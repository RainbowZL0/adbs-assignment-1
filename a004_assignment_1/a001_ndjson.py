import pprint

from a004_assignment_1.a000_CFG import (
    RAW_DATA_FOLDER, NDJSON_FILE_NAME_TO_LOAD,
    FILTERED_DATA_FOLDER,
    SIZE,
    RANK,
    TEST_DATA_FOLDER,
    COMM, NDJSON_TOTAL_LINE_NUM, PIECES_DATA_FOLDER,
)
from a004_assignment_1.a002_utils import (
    write_data_to_ndjson,
    load_ndjson_file_multi_lines_to_list,
    aggregate_score_by_hour,
    split_list,
    join_dict_pieces_hour_score,
    load_ndjson_file_by_process,
    measure_time,
    mpi_v3_subprocess,
    split_file,
    mpi_v4_subprocess, find_the_top_k_v1,
)


def start_one_core():
    records: list = load_ndjson_file_multi_lines_to_list(
        ndjson_path_for_loading=RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD,
        use_filter=True,
    )
    write_data_to_ndjson(
        records=records,
        target_path=FILTERED_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD,
        if_dict_is_single_dict=None,
    )
    pprint.pprint(aggregate_score_by_hour(records))


def mpi_v1():
    """主进程读取全部数据，然后分发给工作进程"""
    if RANK == 0:
        records: list | None = load_ndjson_file_multi_lines_to_list(
            ndjson_path_for_loading=RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD,
            use_filter=True,
        )
        print(f"1. rank={RANK}, 节点读取数据结束")

        records = split_list(lst=records, pieces_num=SIZE)
        print(f"2. rank={RANK}, 数据分片结束")
    else:
        records = None

    received_msg = COMM.scatter(records, root=0)
    print("3. rank={RANK}, 分发数据结束")

    hour_score = aggregate_score_by_hour(received_msg)
    print("4. rank={RANK}, 节点统计数据结束")

    all_hour_score = COMM.gather(hour_score, root=0)
    print("5. rank={RANK}, gather结束")

    if RANK == 0:
        merged_score: dict = join_dict_pieces_hour_score(
            all_hour_score,
            value_type="scalar",
            mode="sum",
        )
        print("6. rank={RANK}, 汇总结束")

        write_data_to_ndjson(
            records=merged_score,
            target_path=TEST_DATA_FOLDER / "gathered.ndjson",
            if_dict_is_single_dict=False,
        )
        print(f"7. rank={RANK}, 保存结果到磁盘结束")


def mpi_v2():
    """所有进程同时开始读取数据"""
    ndjson_path = RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD
    ndjson_line_num = NDJSON_TOTAL_LINE_NUM

    records = load_ndjson_file_by_process(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=SIZE,
        r=RANK,
        use_filter=True,
    )
    # write_data_to_ndjson(
    #     records=records,
    #     target_path=FILTERED_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD,
    # )
    print(f"1. rank={RANK}, 节点读取数据结束")

    hour_score = aggregate_score_by_hour(records)
    print(f"2. rank={RANK}, 节点分别统计结束")

    all_hour_score = COMM.gather(hour_score, root=0)
    print(f"3. rank={RANK}, gather结束")

    if RANK == 0:
        merged_score = join_dict_pieces_hour_score(
            all_hour_score,
            value_type="scalar",
            mode="sum",
        )
        print(f"4. rank={RANK}, 在主节点汇总结束")

        write_data_to_ndjson(
            records=merged_score,
            target_path=TEST_DATA_FOLDER / "gathered_v2.ndjson",
            if_dict_is_single_dict=False,
        )
        print(f"5. rank={RANK}, 保存结果到磁盘结束")


def mpi_v3():
    """所有进程同时开始读取数据，且读取过程中就计算分数"""
    ndjson_path = RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD
    ndjson_line_num = NDJSON_TOTAL_LINE_NUM

    hour_score, failure_records = mpi_v3_subprocess(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=SIZE,
        r=RANK,
        use_filter=False,
    )
    print(f"1. rank={RANK}, 节点读取数据并统计，结束")

    all_hour_score = COMM.gather(hour_score, root=0)
    print(f"2. rank={RANK}, gather分数结束")

    all_failure_records = COMM.gather(failure_records, root=0)
    print(f"3. rank={RANK}, gather failure records结束")

    if RANK == 0:
        merged_score: dict = join_dict_pieces_hour_score(
            all_hour_score,
            value_type="scalar",
            mode="sum",
        )
        print(f"3. rank={RANK}, 在主节点统计分数结束")

        # 4.2 整理所有失败记录
        merged_failures = []
        for sub_failures in all_failure_records:
            merged_failures.extend(sub_failures)
        print(f"4. Rank=0: 收集到 {len(merged_failures)} 条异常记录")

        write_data_to_ndjson(
            records=merged_score,
            target_path=TEST_DATA_FOLDER / "gathered_v3.ndjson",
            if_dict_is_single_dict=False,
        )
        print(f"5. rank={RANK}, 保存分数到磁盘结束")

        write_data_to_ndjson(
            records=merged_failures,
            target_path=TEST_DATA_FOLDER / "failure_v3.ndjson",
            if_dict_is_single_dict=None,
        )
        print(f"6. rank={RANK}, 保存failure到磁盘结束")


def mpi_v4():
    """
    mpi_v4():
    使用事先拆分好的 8 个小 NDJSON 文件（通过 split_file() 产生），
    每个进程借助 mpi_v4_subprocess() 边读取、边计算小时级情感分数，
    最后在 0 号进程上合并并写入最终结果。
    """

    # 1. 根据 RANK 构造拆分文件名称
    split_file_name = f"{NDJSON_FILE_NAME_TO_LOAD.split('.')[0]}_piece_{RANK}.ndjson"
    split_file_path = PIECES_DATA_FOLDER / split_file_name

    # 2. 调用辅助函数进行“边读取边计算”
    hour_score, id_score, failed_records = mpi_v4_subprocess(
        file_path=split_file_path,
        use_filter=False  # 如果想对记录做字段筛选，可以传 True
    )
    print(
        f"Rank={RANK}: "
        f"从文件 {split_file_name} 中解析并计算得到 {len(hour_score)} 个 '小时' 键值对，"
        f"失败 {len(failed_records)} 条"
    )

    # 3. 分别 gather hour_score 和 failed_records
    all_hour_scores = COMM.gather(hour_score, root=0)
    all_id_scores = COMM.gather(id_score, root=0)
    all_failed_records = COMM.gather(failed_records, root=0)

    # 4. 在 0 号进程进行最终合并与输出
    if RANK == 0:
        # 4.1 合并分数
        merged_hour_score: dict = join_dict_pieces_hour_score(
            all_hour_scores,
            value_type="scalar",
            mode="sum",
        )
        print(f"Rank=0: 最终合并后的小时分数 dict 包含 {len(merged_hour_score)} 个小时键")

        merged_id_score = join_dict_pieces_hour_score(
            all_id_scores,
            value_type="list",
            mode="sum",
        )
        print(f"Rank=0: id_score merged")

        # 4.2 整理所有失败记录
        merged_failures: list = []
        for sub_failures in all_failed_records:
            merged_failures.extend(sub_failures)
        print(f"Rank=0: 收集到 {len(merged_failures)} 条异常记录")

        # 4.3 写出最终结果和失败记录
        write_data_to_ndjson(
            records=merged_hour_score,
            target_path=TEST_DATA_FOLDER / "merged_hour_score_v4.ndjson",
            if_dict_is_single_dict=False,
        )
        write_data_to_ndjson(
            records=merged_id_score,
            target_path=TEST_DATA_FOLDER / "merged_id_score_v4.ndjson",
            if_dict_is_single_dict=False,
        )
        write_data_to_ndjson(
            records=merged_failures,
            target_path=TEST_DATA_FOLDER / "merged_failures_v4.ndjson",
            if_dict_is_single_dict=None,
        )

        print(
            "写入完毕"
        )


def measure_mpi(func):
    if RANK == 0:
        print(measure_time(func)())
    else:
        func()


def call_split_file():
    split_file(
        file_path=RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD,
        total_line_num=NDJSON_TOTAL_LINE_NUM,
        to_pieces_num=8,
        output_folder=PIECES_DATA_FOLDER,
        use_filter=True,
    )


def call_find_top_k(file_path_for_loading, top_k, is_max):
    records: list = load_ndjson_file_multi_lines_to_list(
        ndjson_path_for_loading=file_path_for_loading,
        use_filter=False
    )
    top: list = find_the_top_k_v1(record, top_k=top_k, is_max=is_max)
    pprint.pprint(top)


if __name__ == "__main__":
    # mpi_v1()
    measure_mpi(mpi_v4)
    # call_find_top_k(
    #     TEST_DATA_FOLDER / "gathered_v4.ndjson",
    #     top_k=10,
    #     is_max=False,
    # )
    print("Process finished.")
