import functools
import json
import pprint
import traceback
from datetime import datetime
from math import ceil
from pathlib import Path

from mpi4py import MPI
from tqdm import tqdm


def load_ndjson_file(
        ndjson_path_for_loading: str | Path,
        use_filter: bool = False,
) -> list:
    """从 NDJSON 文件中读取数据。

        Args:
            ndjson_path_for_loading (str | Path): NDJSON 文件路径。
            use_filter (bool, 可选): 是否对每条记录进行过滤。默认为 False。

        Returns:
            list: 读取到的记录列表。
    """
    records: list = []
    with open(ndjson_path_for_loading, "r", encoding="utf-8") as f:
        for line in f:
            record = parse_one_line(line, use_filter=use_filter)
            records.append(record)
    return records


def parse_one_line(line, use_filter):
    if not line:
        return
    line = line.strip()
    record = json.loads(line)
    if use_filter:
        record = filter_a_record(record)
    return record


def dict_to_a_line(dic):
    line = json.dumps(dic, ensure_ascii=False)
    return f"{line}\n"


def filter_a_record(record: dict):
    """对单条记录进行字段筛选，提取关键信息。

    Args:
        record (dict): 原始记录字典。

    Returns:
        dict: 筛选后的记录，仅保留指定字段。
    """
    doc = record.get("doc", {})
    account = doc.get("account", {})
    return {
        "doc": {
            "createdAt": doc.get("createdAt"),
            "sentiment": doc.get("sentiment"),
            "account": {
                key: account.get(key)
                for key in (
                    "id",
                    "username",
                    # "acct",
                    # "uri",
                    # "url",
                    # "displayName",
                )
            },
        }
    }


def write_data_to_ndjson(
        records: list | dict,
        target_path: str | Path,
):
    """将数据写入 NDJSON 文件。

    Args:
        records (list | dict): 要写入的记录（单条或多条）。
        target_path (str | Path): 目标文件路径。
    """
    if isinstance(records, dict):
        records = [records]
    with open(target_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                dict_to_a_line(record)
            )


def high_level_api_to_filter_ndjson_and_save(
        ndjson_path_for_loading: str | Path,
        ndjson_path_for_saving: str | Path,
):
    """高级封装：读取 NDJSON 文件，筛选字段后保存到新文件。

    Args:
        ndjson_path_for_loading (str | Path): 原始 NDJSON 文件路径。
        ndjson_path_for_saving (str | Path): 筛选后保存的新 NDJSON 文件路径。

    Returns:
        list: 筛选后的记录列表。
    """
    records = load_ndjson_file(
        ndjson_path_for_loading=ndjson_path_for_loading,
        use_filter=True,
    )
    write_data_to_ndjson(records=records, target_path=ndjson_path_for_saving)
    return records


def high_level_api_to_convert_raw_time_to_preferred_str(t):
    """将原始时间字符串转换为“年月日 小时”格式字符串。

    Args:
        t (str): 原始 ISO 8601 时间字符串。

    Returns:
        str: 格式化后的时间字符串（精确到小时）。
    """
    return iso_format_time_to_str(
        floor_time_to_the_latest_hour(raw_time_to_py_datetime(t))
    )


def raw_time_to_py_datetime(t):
    """将 ISO 格式的字符串时间转换为 Python datetime 对象。

    Args:
        t (str): 原始时间字符串，可能以 'Z' 结尾。

    Returns:
        datetime: Python datetime 对象。
    """
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def floor_time_to_the_latest_hour(t):
    """将时间向下舍入至整点小时。

    Args:
        t (datetime): 原始时间对象。

    Returns:
        datetime: 舍入到整点小时的时间。
    """
    return t.replace(minute=0, second=0, microsecond=0)


def iso_format_time_to_str(t):
    """将 datetime 对象格式化为字符串（YYYY-MM-DD HH:MM）。

    Args:
        t (datetime): 时间对象。

    Returns:
        str: 格式化后的时间字符串。
    """
    return t.strftime("%Y-%m-%d %H:%M")


def aggregate_score_by_hour(records: list) -> dict:
    """按小时聚合情感得分。

    Args:
        records (list): 包含时间和情感得分的记录列表。

    Returns:
        dict: 键为小时（字符串），值为对应小时的总得分。
    """
    time_s_score: dict = {}
    for record in records:
        created_hour: str = high_level_api_to_convert_raw_time_to_preferred_str(
            record["doc"]["createdAt"]
        )
        sentiment_score = record["doc"]["sentiment"]
        # print(created_hour)

        if not time_s_score.get(created_hour):
            time_s_score[created_hour] = 0.0
        time_s_score[created_hour] += sentiment_score
    # """
    # change structure,
    # from {
    #     'hour': score,
    #     'hour': score,
    # }
    # to [
    #     {'hour': score},
    #     {'hour': score},
    # ]
    # """
    # return [{k: v} for k, v in time_s_score.items()]
    return time_s_score


def get_ndjson_name_by_rank(original_name, r):
    """根据原始文件名和编号生成新的带编号的文件名。

    Args:
        original_name (str | Path): 原始文件名。
        r (int): 编号（用于区分分片）。

    Returns:
        str: 带编号的新文件名。
    """
    if not isinstance(original_name, Path):
        new_name = Path(original_name)
    else:
        new_name = original_name
    return new_name.stem + f"_r-{r}" + new_name.suffix


def split_list(lst, pieces_num):
    """将列表分割为若干个子列表。

    如果不能平均分割，则最后一个子列表元素可能较少。

    Args:
        lst (list): 原始列表。
        pieces_num (int): 分成的子列表数量。

    Returns:
        list: 分割后的子列表组成的列表。
    """
    result = []
    elem_per_piece = ceil(len(lst) / pieces_num)
    for i in range(pieces_num):
        start = i * elem_per_piece
        end = min(start + elem_per_piece, len(lst))
        result.append(lst[start:end])
    return result


def join_list_pieces(lst):
    """将多个子列表合并为一个完整列表。

    Args:
        lst (list): 子列表组成的列表。

    Returns:
        list: 合并后的完整列表。
    """
    result = lst[0]
    for i in range(1, len(lst)):
        result.extend(lst[i])
    return result


def join_dict_pieces_hour_score(lst, mode="sum"):
    """合并多个字典（按 key 聚合 value）。

    Args:
        lst (list): 字典组成的列表。
        mode (str, 可选): 聚合方式，目前支持 "sum"。默认为 "sum"。

    Returns:
        dict: 合并后的字典。
    """
    result = {}
    for dic in lst:
        for k, v in dic.items():
            if not result.get(k):
                result[k] = v
            else:
                if mode == "sum":
                    result[k] += v
    return result


def dict_to_list_of_tuples(dic):
    """将字典转换为 (key, value) 元组列表。

    Args:
        dic (dict): 输入字典。

    Returns:
        list: (key, value) 形式的元组列表。
    """
    return [(k, v) for k, v in dic.items()]


def load_ndjson_file_by_process(
        ndjson_path_for_loading,
        ndjson_line_num,
        process_num,
        r,
        use_filter=False,
):
    num_line_per_process = ceil(ndjson_line_num / process_num)
    # [start, end) is retrieved, and the first line of the file is counted from 1
    start_line = r * num_line_per_process + 1
    end_line = min(start_line + num_line_per_process, int(ndjson_line_num + 1))

    records = []
    with open(ndjson_path_for_loading, "r", encoding="utf-8") as f0:
        # 跳过前面的行
        for _ in range(start_line - 1):
            next(f0)
        # 读取所需行
        for _ in range(start_line, end_line):
            line = next(f0)
            record: dict = parse_one_line(line, use_filter=use_filter)
            records.append(record)

    return records


def load_ndjson_file_by_process_and_calcu_score_at_the_same_time(
        ndjson_path_for_loading,
        ndjson_line_num,
        process_num,
        r,
        use_filter=False,
):
    num_line_per_process = ceil(ndjson_line_num / process_num)
    # [start, end) is retrieved, and the first line of the file is counted from 1
    start_line = r * num_line_per_process + 1
    end_line = min(start_line + num_line_per_process, int(ndjson_line_num + 1))

    time_s_score: dict = {}
    failed_records = []
    with open(ndjson_path_for_loading, "r", encoding="utf-8") as f0:
        # 跳过前面的行
        for _ in range(start_line - 1):
            next(f0)
        # 读取所需行
        for i in range(start_line, end_line):
            line = next(f0)
            record = parse_one_line(line, use_filter=use_filter)

            """
            直接清洗
            格式化时间，提取分数
            """
            try:
                created_hour: str = high_level_api_to_convert_raw_time_to_preferred_str(
                    record["doc"]["createdAt"]
                )
                sentiment_score = record["doc"]["sentiment"]

                if not time_s_score.get(created_hour):
                    time_s_score[created_hour] = 0.0
                time_s_score[created_hour] += sentiment_score
            except Exception as e:
                print(f"At line {i}, an exception occurred, {e}:")
                traceback.print_exc()
                pprint.pprint(record)
                failed_records.append(record)
                continue

    return time_s_score, failed_records


def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = MPI.Wtime()
        result = func(*args, **kwargs)
        end = MPI.Wtime()
        running_info = f"{func.__name__}: {round(end - start, 5)}"
        return result, running_info

    return wrapper


def split_file(
        file_path,
        total_line_num,
        to_pieces_num,
        output_folder,
        use_filter=False
):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if not isinstance(output_folder, Path):
        output_folder = Path(output_folder)

    lines_per_file = ceil(total_line_num / to_pieces_num)
    with open(file_path, "r", encoding="utf-8") as f0:
        for i in tqdm(range(to_pieces_num)):
            if i < to_pieces_num - 1:
                lines_for_file_i = lines_per_file
            else:
                lines_for_file_i = total_line_num - i * lines_per_file

            output_filename = f"{file_path.stem}_piece_{i}{file_path.suffix}"
            output_path = output_folder / output_filename

            with open(output_path, "w", encoding="utf-8") as f1:
                for j in tqdm(range(lines_for_file_i)):
                    line = next(f0)
                    try:
                        record = parse_one_line(line, use_filter=use_filter)
                        write_line = dict_to_a_line(record)
                        f1.write(write_line)
                    except Exception as e:
                        cur_line_num = i * lines_per_file + j + 1
                        print(f"At line {cur_line_num}, an exception occurred, {e}:")
                        traceback.print_exc()
                        pprint.pprint(record)
                        continue


def tst():
    for _ in tqdm(range(int(1e8))):
        pass


if __name__ == '__main__':
    # load_ndjson_file_by_process()
    tst()
