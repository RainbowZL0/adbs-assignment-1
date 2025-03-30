import heapq
import pprint

from a004_assignment_1.a000_CFG import TEST_DATA_FOLDER
from a004_assignment_1.a002_utils import parse_one_line
from a004_assignment_1.a004_comparing_obj import ComparableObj


def find_the_top_k_v1(comparable_obj_gnr, top_k, get_max=True):
    heap = []
    for i, obj in enumerate(comparable_obj_gnr):
        obj: ComparableObj
        if not get_max:
            obj.update_v(-obj.v)
        if i < top_k:
            heapq.heappush(heap, obj)
        else:
            if obj > heap[0]:
                heapq.heapreplace(heap, obj)
    heap.sort(reverse=True)

    if not get_max:
        for i in range(len(heap)):
            heap[i].update_v(-heap[i].v)

    return heap


def find_the_top_k_v2(
        tuple_gnr,
        top_k,
        get_max=True,
):
    if get_max:
        # 直接取前 K 个“最大”
        return heapq.nlargest(
            top_k,
            tuple_gnr,
            key=lambda x: x[0],
        )
    else:
        # 直接取前 K 个“最小”
        return heapq.nsmallest(
            top_k,
            tuple_gnr,
            key=lambda x: x[0],
        )


def list_of_comparable_obj_to_list_of_dict(
        lst: list[ComparableObj]
):
    return [item.get_dict() for item in lst]


def load_ndjson_and_find_the_top_k_v1(
        input_ndjson_path,
        top_k,
        value_type,
        get_max,
        use_filter,
):
    """
    使用自定义的get_gnr_comparable_obj，使用自定义的堆结构
    """
    dict_gnr = get_dict_gnr_read_ndjson_file_multi_lines(
        input_ndjson_path=input_ndjson_path,
        use_filter=use_filter,
    )
    comparable_obj_gnr = get_gnr_comparable_obj(
        dict_gnr=dict_gnr,
        value_type=value_type,
    )
    lst_of_comparable_obj = find_the_top_k_v1(
        comparable_obj_gnr=comparable_obj_gnr,
        top_k=top_k,
        get_max=get_max,
    )
    lst_of_dict = list_of_comparable_obj_to_list_of_dict(lst_of_comparable_obj)
    return lst_of_dict


def load_ndjson_and_find_the_top_k_v2(
        input_ndjson_path,
        top_k,
        value_type,
        get_max,
        use_filter,
):
    """
    使用最佳实践，在(key, obj)上执行比较；
    使用内置的堆结构。
    """
    dict_gnr = get_dict_gnr_read_ndjson_file_multi_lines(
        input_ndjson_path=input_ndjson_path,
        use_filter=use_filter,
    )
    comparable_tuple_gnr = get_gnr_comparable_tuple(
        dict_gnr=dict_gnr,
        value_type=value_type,
    )
    lst_of_comparable_tuple = find_the_top_k_v2(
        tuple_gnr=comparable_tuple_gnr,
        top_k=top_k,
        get_max=get_max,
    )
    lst_of_dict = list_of_comparable_tuple_to_list_of_original_data(
        lst_of_comparable_tuple
    )
    return lst_of_dict


def get_comparable_tuple(dic, value_type):
    key = list(dic)[0]
    if value_type == "scalar":
        return dic[key], dic
    else:
        return dic[key][0], dic


def get_value_of_obj_being_compared(obj, version):
    if version == 1:
        return obj.v
    else:
        return obj[0]


def get_gnr_comparable_tuple(dict_gnr, value_type):
    for item in dict_gnr:
        yield get_comparable_tuple(item, value_type)


def list_of_comparable_tuple_to_list_of_original_data(lst):
    _, rst = zip(*lst)
    return list(rst)


def get_dict_gnr_read_ndjson_file_multi_lines(
        input_ndjson_path,
        use_filter=False,
):
    """迭代器，每次返回读取一行得到的dict
    Args:
        input_ndjson_path:
        use_filter:

    Returns:
        dict:
    """
    with open(input_ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            yield parse_one_line(
                line,
                use_filter=use_filter,
            )


def get_gnr_comparable_obj(dict_gnr, value_type):
    for i in dict_gnr:
        yield ComparableObj(i, value_type)


def tst():
    rst = load_ndjson_and_find_the_top_k_v2(
        input_ndjson_path=TEST_DATA_FOLDER / "top_k_test_2.ndjson",
        top_k=10,
        value_type="list",
        get_max=False,
        use_filter=False,
    )
    pprint.pprint(rst)


def tst_list_of_comparable_tuple_to_list_of_original_data():
    a = [(1, '1'), (2, '2'), (3, '3'), (4, '4')]
    print(list_of_comparable_tuple_to_list_of_original_data(a))


if __name__ == '__main__':
    tst()
    pass
