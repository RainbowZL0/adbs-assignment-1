from a004_assignment_1.a000_CFG import RAW_DATA_FOLDER, NDJSON_FILE_NAME_TO_LOAD, NDJSON_TOTAL_LINE_NUM
from a004_assignment_1.a002_utils import load_ndjson_file_by_process


# noinspection PyUnusedLocal
def tst_load_ndjson_file_by_process():
    ndjson_path = RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD
    ndjson_line_num = NDJSON_TOTAL_LINE_NUM

    records_0 = load_ndjson_file_by_process(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=4,
        r=0,
        use_filter=True,
    )
    records_1 = load_ndjson_file_by_process(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=4,
        r=1,
        use_filter=True,
    )
    records_2 = load_ndjson_file_by_process(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=4,
        r=2,
        use_filter=True,
    )
    records_3 = load_ndjson_file_by_process(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=4,
        r=3,
        use_filter=True,
    )
    pass


def tst():
    dic = {
        '1': 1,
        '2': 2,
    }
    a = len(dic)
    print(a)


if __name__ == '__main__':
    tst()
    pass
