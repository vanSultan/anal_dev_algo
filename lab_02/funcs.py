from math import sin
from random import random, normalvariate, seed
from typing import Callable, Tuple, List

from vector_2d import Vector
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
        x += eps

    return (x_min, y_min), f_min


def gauss_method(
        fun: Callable[[type_approx_func, type_signal, float, float], float], fun_sub: type_approx_func,
        signal: type_signal, limits: Tuple[Tuple[float, float], Tuple[float, float]], eps: float = 1e-3
) -> Tuple[Tuple[float, float], float]:
    vec_par, f_min, f_min_prev = list(limits[0]), None, None
    k, cnt_par = 0, len(vec_par)
    while True:
        vec_cur = vec_par.copy()
        vec_cur[k] = limits[0][k]
        while vec_cur[k] <= limits[1][k]:
            f_cur = fun(fun_sub, signal, *vec_cur)
            if f_min is None or f_cur < f_min:
                vec_par[k], f_min = vec_cur[k], f_cur
            vec_cur[k] += eps
        k = (k + 1) % cnt_par
        if f_min_prev is None or abs(f_min - f_min_prev) > eps:
            f_min_prev = f_min
        else:
            break

    return (vec_par[0], vec_par[1]), f_min


def nelder_mead(
        fun: Callable[[type_approx_func, type_signal, float, float], float], fun_sub: type_approx_func,
        signal: type_signal, limits: Tuple[Tuple[float, float], Tuple[float, float]], eps: float = 1e-3,
        alpha: float = 1., beta: float = 0.5, gamma: float = 2.
) -> Tuple[Tuple[float, float], float]:
    v_1, v_2, v_3 = Vector(0., 0.), Vector(1., 0.), Vector(0., 1.)
    f_min = None
    while True:
        dic = {
            v_1: fun(fun_sub, signal, *v_1.get_comps()),
            v_2: fun(fun_sub, signal, *v_2.get_comps()),
            v_3: fun(fun_sub, signal, *v_3.get_comps())
        }
        p_lst = list(map(lambda el: el[0], sorted(dic.items(), key=lambda el: el[1])))

        middle = (p_lst[0] + p_lst[1]) / 2
        xr = middle + alpha * (middle - p_lst[2])
        f_xr = fun(fun_sub, signal, *xr.get_comps())
        if f_xr < dic[p_lst[1]]:
            p_lst[2] = xr
        else:
            if f_xr < dic[p_lst[2]]:
                p_lst[2] = xr
                dic[xr] = f_xr
            c = (p_lst[2] + middle) / 2
            f_c = fun(fun_sub, signal, *c.get_comps())
            if f_c < dic[p_lst[2]]:
                p_lst[2] = c
                dic[c] = f_c
        if f_xr < dic[p_lst[0]]:
            xe = middle + gamma * (xr - middle)
            f_xe = fun(fun_sub, signal, *xe.get_comps())
            if f_xe < f_xr:
                p_lst[2] = xe
                dic[xe] = f_xe
            else:
                p_lst[2] = xr
                dic[xr] = f_xr
        if f_xr > dic[p_lst[1]]:
            xc = middle + beta * (p_lst[2] - middle)
            f_xc = fun(fun_sub, signal, *xc.get_comps())
            if f_xc < dic[p_lst[2]]:
                p_lst[2] = xc
                dic[xc] = f_xc

        v_1, v_2, v_3 = p_lst[2], p_lst[1], p_lst[0]

        if f_min is None or abs(f_min - dic[v_3]) > eps:
            f_min = dic[v_3]
        else:
            break
    return v_3.get_comps(), dic[v_3]


if __name__ == '__main__':
    seed()
    s = get_signal()
    xx = list(map(lambda i: i[0], s))
    yy = list(map(lambda i: i[1], s))

    fig, ax = plt.subplots(2, 1)

    ax[0].scatter(xx, yy, color="deeppink")
    ax[1].scatter(xx, yy, color="deeppink")

    lim = ((min(xx), min(yy)), (max(xx), max(yy)))
    # lim = ((0., -0.5), (1., 0.5))
    print(lim)

    c_1, _ = exhaustive_search_2d(least_squares, linear_approx, s, lim)
    print(c_1)
    c_2, _ = gauss_method(least_squares, linear_approx, s, lim)
    print(c_2)
    c_3, _ = nelder_mead(least_squares, linear_approx, s, lim)
    print(c_3)
    ax[0].plot(xx, [linear_approx(k, c_1[0], c_1[1]) for k in xx], color="c")
    ax[0].plot(xx, [linear_approx(k, c_2[0], c_2[1]) for k in xx], color="m")
    ax[0].plot(xx, [linear_approx(k, c_3[0], c_3[1]) for k in xx], color="r")

    c_4, _ = exhaustive_search_2d(least_squares, rational_approx, s, lim)
    print(c_4)
    c_5, _ = gauss_method(least_squares, rational_approx, s, lim)
    print(c_5)
    c_6, _ = nelder_mead(least_squares, rational_approx, s, lim)
    print(c_6)
    ax[1].plot(xx, [rational_approx(k, c_4[0], c_4[1]) for k in xx], color="c")
    ax[1].plot(xx, [rational_approx(k, c_5[0], c_5[1]) for k in xx], color="m")
    ax[1].plot(xx, [rational_approx(k, c_6[0], c_6[1]) for k in xx], color="r")

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
