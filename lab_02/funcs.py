from math import sin
from random import random, normalvariate
from typing import Callable, Tuple, List

from matplotlib import pyplot as plt


def function_1(x: float) -> float:
    # [0, 1]
    return x ** 3


def function_2(x: float) -> float:
    # [0, 1]
    return abs(x - 0.2)


def function_3(x: float) -> float:
    # [0.1, 1]
    return x * sin(1 / x)


def exhaustive_search(fun: Callable[[float], float], limits: Tuple[float, float],
                      eps: float = 1e-3) -> Tuple[float, float]:
    x_min, f_min, x = None, None, limits[0]

    while x <= limits[1]:
        f_cur = fun(x)
        if x_min is None or f_cur < f_min:
            x_min, f_min = x, f_cur
        x += eps

    n = round((limits[1] - limits[0]) / eps)
    print(f"Cnt of fun eval: {n}\nCnt of iter: {n}")

    return x_min, f_min


def dichotomy_method(fun: Callable[[float], float], limits: Tuple[float, float],
                     eps: float = 1e-3) -> Tuple[float, float]:
    delta = eps / 10
    a, b = limits
    x_1, x_2 = (a + b - delta) / 2, (a + b + delta) / 2
    iter_cnt, fun_call_cnt = 0, 0

    while abs(b - a) >= eps:
        f_1, f_2 = fun(x_1), fun(x_2)
        if f_1 <= f_2:
            b = x_2
        else:
            a = x_1
        x_1, x_2 = (a + b - delta) / 2, (a + b + delta) / 2
        iter_cnt, fun_call_cnt = iter_cnt + 1, fun_call_cnt + 2
    x_min = (x_1 + x_2) / 2

    print(f"Cnt of fun eval: {fun_call_cnt}\nCnt of iter: {iter_cnt}")

    return x_min, fun(x_min)


def golden_section(fun: Callable[[float], float], limits: Tuple[float, float],
                   eps: float = 1e-3) -> Tuple[float, float]:
    a, b = limits
    k = (pow(5, 0.5) - 1) / 2
    x_1, x_2 = a + (1 - k) * (b - a), a + k * (b - a)
    f_1, f_2 = fun(x_1), fun(x_2)
    iter_cnt, fun_call_cnt = 0, 2

    while abs(b - a) >= eps:
        if f_1 > f_2:
            a = x_1
            x_1 = x_2
            x_2 = a + k * (b - a)
            f_1 = f_2
            f_2 = fun(x_2)
        else:
            b = x_2
            x_2 = x_1
            x_1 = a + (1 - k) * (b - a)
            f_2 = f_1
            f_1 = fun(x_1)
        iter_cnt, fun_call_cnt = iter_cnt + 1, fun_call_cnt + 1
    x_min = (a + b) / 2

    print(f"Cnt of fun eval: {fun_call_cnt}\nCnt of iter: {iter_cnt}")

    return x_min, fun(x_min)


type_approx_func = Callable[[float, float, float], float]
type_signal = List[Tuple[float, float]]


def get_signal(n: int = 101) -> List[Tuple[float, float]]:
    alpha, beta = random(), random()
    signal = [(x, alpha * x + beta + delta) for x, delta in [(k / 100., normalvariate(0, 1)) for k in range(n)]]

    return signal


def linear_approx(x: float, a: float, b: float) -> float:
    return a * x + b


def rational_approx(x: float, a: float, b: float) -> float:
    return a / (1 + b * x)


def least_squares(fun: type_approx_func, signal: type_signal, a: float, b: float) -> float:
    d = sum(pow(fun(x, a, b) - y, 2) for x, y in signal)
    return d


def exhaustive_search_2d(
        fun: Callable[[type_approx_func, type_signal, float, float], float], fun_sub: type_approx_func,
        signal: type_signal, limits: Tuple[Tuple[float, float], Tuple[float, float]], eps: float = 1e-3
) -> Tuple[Tuple[float, float], float]:
    x = limits[0][0]
    x_min, y_min, f_min = None, None, None
    while x <= limits[1][0]:
        y = limits[0][1]
        while y <= limits[1][1]:
            f_cur = fun(fun_sub, signal, x, y)
            if f_min is None or f_cur < f_min:
                x_min, y_min, f_min = x, y, f_cur
            y += eps
            i += 1
        x += eps

    return (x_min, y_min), f_min


def gauss_method(
        fun: Callable[[type_approx_func, type_signal, float, float], float], fun_sub: type_approx_func,
        signal: type_signal, limits: Tuple[Tuple[float, float], Tuple[float, float]], eps: float = 1e-3
) -> Tuple[Tuple[float, float], float]:
    init_ap = list(limits[0])
    vec_min, f_min = init_ap.copy(), None
    k = 0
    while True:
        while init_ap[k] <= limits[1][k]:
            f_cur = fun(fun_sub, signal, *init_ap)
            if f_min is None or f_cur < f_min:
                vec_min[k], f_min = init_ap[k], f_cur
            init_ap[k] += eps
        init_ap[k] = limits[0][k]
        k = (k + 1) % len(init_ap)


if __name__ == '__main__':
    s = get_signal()
    xx = list(map(lambda i: i[0], s))
    yy = list(map(lambda i: i[1], s))
    lim = ((min(xx), min(yy)), (max(xx), max(yy)))
    print(lim)
    c_1, _ = exhaustive_search_2d(least_squares, linear_approx, s, lim)
    print(c_1)
    c_2, _ = exhaustive_search_2d(least_squares, rational_approx, s, lim)
    print(c_2)
    plt.scatter(xx, yy, color="deeppink")
    plt.plot(xx, [linear_approx(k, c_1[0], c_1[1]) for k in xx], color="c")
    plt.plot(xx, [rational_approx(k, c_2[0], c_2[1]) for k in xx], color="m")
    plt.show()
    # ep = 1e-3
    #
    # print("fun_1")
    # print(exhaustive_search(function_1, (0, 1), ep))
    # print(dichotomy_method(function_1, (0, 1), ep))
    # print(golden_section(function_1, (0, 1), ep))
    # print()
    # print("fun_2")
    # print(exhaustive_search(function_2, (0, 1), ep))
    # print(dichotomy_method(function_2, (0, 1), ep))
    # print(golden_section(function_2, (0, 1), ep))
    # print()
    # print("fun_3")
    # print(exhaustive_search(function_3, (0.1, 1), ep))
    # print(dichotomy_method(function_3, (0.1, 1), ep))
    # print(golden_section(function_3, (0.1, 1), ep))
