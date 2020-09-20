from functools import reduce


def constant_function(v: list, c: float = 1.) -> float:
    return c


def sum_function(v: list) -> float:
    return sum(v)


def prod_function(v: list) -> float:
    return reduce(lambda a, b: a * b, v)


def naive_poly_function(v: list, x: float = 1.5) -> float:
    return sum([val * pow(x, k) for k, val in enumerate(v)])


def gorner_method_function(v: list, x: float = 1.5) -> float:
    return reduce(lambda agg, a: a + x * agg, reversed(v), 0)


def bubble_sort(v: list) -> list:
    for i in range(len(v) - 1):
        for j in range(len(v) - i - 1):
            if v[j] > v[j + 1]:
                v[j], v[j + 1] = v[j + 1], v[j]

    return v


def quick_sort(v: list) -> list:
    def partition(lst: list, lw: int, hg: int) -> int:
        i, j = lw, hg + 1
        while True:
            while True:
                i += 1
                if not(lst[i] < lst[lw]) or i == hg:
                    break
            while True:
                j -= 1
                if not(lst[j] > lst[lw]) or j == lw:
                    break
            if i >= j:
                break
            lst[i], lst[j] = lst[j], lst[i]
        lst[lw], lst[j] = lst[j], lst[lw]
        return j

    def quick_sort_routine(lst: list, lw: int, hg: int) -> None:
        if hg <= lw:
            return
        j = partition(lst, lw, hg)
        quick_sort_routine(lst, lw, j - 1)
        quick_sort_routine(lst, j + 1, hg)

    quick_sort_routine(v, 0, len(v) - 1)
    return v


if __name__ == '__main__':
    print(bubble_sort([3, 2, 1]))
    print(quick_sort([3, 2, 1]))
