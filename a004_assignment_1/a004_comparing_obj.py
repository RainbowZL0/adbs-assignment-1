from functools import total_ordering


class ComparableObj:
    @total_ordering
    def __init__(self, dic, value_type):
        """
        Args:
            dic (dict):
                - {str hour: float v}
                - or {str id: [float v, str username]}
        """
        assert value_type in ["scalar", "list"]
        assert len(dic) == 1

        self.value_type = value_type
        self.dic = dic
        self.k = list(dic)[0]

        if value_type == "scalar":
            self.v = dic[self.k]
        else:
            self.v = dic[self.k][0]

    def __lt__(self, other):
        self.check_same_value_type(other)
        return self.v < other.v

    def __eq__(self, other):
        self.check_same_value_type(other)
        return self.v == other.v

    def check_same_value_type(self, other):
        if self.value_type != other.value_type:
            raise TypeError(
                f"value_type {self.value_type} != {other.value_type}"
            )

    def update_v(self, v):
        self.v = v
        if self.value_type == "scalar":
            self.dic[self.k] = v
        else:
            self.dic[self.k][0] = v

    def get_dict(self):
        return self.dic


def tst():
    a = {
        "k_a": 1
    }
    b = {
        "k_b": 2
    }
    a = ComparableObj(a, "scalar")
    b = ComparableObj(b, "scalar")

    r = (a < b)
    print(r)

    a = {
        "k_a": [1, "haha"]
    }
    b = {
        "k_b": [2, "haha"]
    }
    a = ComparableObj(a, "list")
    b = ComparableObj(b, "list")

    r = (a < b)
    print(r)


if __name__ == '__main__':
    tst()
