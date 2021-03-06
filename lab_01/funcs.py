from functools import reduce
from typing import List


def constant_function(v: list, c: float = 1.) -> float:
    return c


def sum_function(v: list) -> float:
    return sum(v)


def prod_function(v: list) -> float:
    return reduce(lambda a, b: a * b, v)


def naive_poly_function(v: list, x: float = 1.5) -> float:
    return sum([val * pow(x, k) for k, val in enumerate(v)])


def horner_method_function(v: list, x: float = 1.5) -> float:
    return reduce(lambda agg, a: a + x * agg, reversed(v), 0)


def bubble_sort(v: list) -> list:
    sort_v = v.copy()
    for i in range(len(sort_v) - 1):
        for j in range(len(sort_v) - i - 1):
            if sort_v[j] > sort_v[j + 1]:
                sort_v[j], sort_v[j + 1] = sort_v[j + 1], sort_v[j]

    return sort_v


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

    sort_v = v.copy()
    quick_sort_routine(sort_v, 0, len(sort_v) - 1)
    return sort_v


def get_minrun(n: int) -> int:
    r = 0
    while n >= 64:
        r |= n & 1
        n >>= 1
    res = n + r
    return res


def insertion_sort(v: list, left: int = 0, right: int = None) -> list:
    if not right:
        right = len(v)
    for i in range(left + 1, right):
        temp = v[i]
        j = i - 1
        while j >= left and v[j] > temp:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = temp

    return v


def merge_runs(v: list, left: int, middle: int, right: int) -> list:
    len_1, len_2 = middle - left + 1, right - middle
    left_lst, right_lst = list(), list()
    for i in range(len_1):
        left_lst.append(v[left + i])
    for i in range(len_2):
        right_lst.append(v[middle + 1 + i])

    i, j, k = 0, 0, left
    while i < len_1 and j < len_2:
        if left_lst[i] <= right_lst[j]:
            v[k] = left_lst[i]
            i += 1
        else:
            v[k] = right_lst[j]
            j += 1
        k += 1
    while i < len_1:
        v[k] = left_lst[i]
        k += 1
        i += 1
    while j < len_2:
        v[k] = right_lst[j]
        k += 1
        j += 1

    return v


def timsort(v: list) -> list:
    sort_v = v.copy()
    n = len(sort_v)
    minrun = get_minrun(n)

    for i in range(0, n, minrun):
        insertion_sort(sort_v, i, min(i + minrun, n))
    size = minrun
    while size < n:
        for left in range(0, n, size * 2):
            middle = left + size - 1
            right = min(left + size * 2 - 1, n - 1)
            merge_runs(sort_v, left, middle, right)
        size *= 2

    return sort_v


def prod_matrix(a: List[list], b: List[list]) -> List[list]:
    c = [[sum(x * y for x, y in zip(a_row, b_col)) for b_col in zip(*b)] for a_row in a]

    return c


def vinograd_mod(a: List[list], b: List[list]) -> List[list]:
    m, n, q = len(a), len(b), len(b[0])
    mul_h, mul_v = [0] * m, [0] * q
    c = [[0 for j in range(len(b[0]))] for i in range(len(a))]

    for i in range(m):
        for j in range(0, n - 1, 2):
            mul_h[i] += a[i][j] * a[i][j + 1]

    for i in range(q):
        for j in range(0, n - 1, 2):
            mul_v[i] += b[j][i] * b[j + 1][i]

    for i in range(m):
        for j in range(q):
            c[i][j] = -(mul_h[i] + mul_v[j])

            for k in range(0, n - 1, 2):
                c[i][j] += (a[i][k] + b[k + 1][j]) * (a[i][k + 1] + b[k][j])

            c[i][j] += a[i][n - 1] * b[n - 1][j] if n % 2 else 0

    return c
