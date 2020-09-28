from random import seed, random
from typing import Callable, Tuple

from matplotlib import pyplot as plt

from lab_02 import *


def show_part_1(begin_info: str, fun: Callable[[float], float], lim: Tuple[float, float]) -> None:
    print(begin_info)
    print("exhaustive_search (x, f): {r[0]:.3f}, {r[1]:.3f}".format(r=exhaustive_search(fun, lim)))
    print("dichotomy_method  (x, f): {r[0]:.3f}, {r[1]:.3f}".format(r=dichotomy_method(fun, lim)))
    print("golden_section    (x, f): {r[0]:.3f}, {r[1]:.3f}".format(r=golden_section(fun, lim)))


if __name__ == '__main__':
    seed()

    show_part_1("[fun_1]: x^3, [0, 1]", function_1, (0., 1.))
    show_part_1("[fun_2]: |x - 0.2|, [0, 1]", function_2, (0., 1.))
    show_part_1("[fun_3]: x*sin(1/x), [0.1, 1]", function_1, (0.1, 1.))

    signal = get_signal()

    x, y = list(map(lambda k: k[0], signal)), list(map(lambda k: k[1], signal))
    limits = ((min(x), min(y)), (max(x), max(y)))

    fig, ax = plt.subplots(2)
    titles = ["Linear approximation", "Rational approximation"]
    funcs = [linear_approx, rational_approx]
    for i in range(2):
        ax[i].set_title(titles[i])
        ax[i].grid(1)
        ax[i].scatter(x, y, color="blue", marker=".", label="signal")

        print(titles[i])
        coefficients_1, f_1 = exhaustive_search_2d(least_squares, funcs[i], signal, limits)
        print(f"exhaustive_search_2d ((a, b), f): ({coefficients_1[0]:.3f}, {coefficients_1[1]:.3f}), {f_1}")
        ax[i].plot(x, [funcs[i](j, coefficients_1[0], coefficients_1[1]) for j in x],
                   color="c", label="exhaustive_search_2d")
        coefficients_2, f_2 = gauss_method(least_squares, funcs[i], signal, limits)
        print(f"gauss_method ((a, b), f): ({coefficients_2[0]:.3f}, {coefficients_2[1]:.3f}), {f_2}")
        ax[i].plot(x, [funcs[i](j, coefficients_2[0], coefficients_2[1]) for j in x], color="m", label="gauss_method")
        coefficients_3, f_3 = nelder_mead(least_squares, funcs[i], signal)
        print(f"nelder_mead ((a, b), f): ({coefficients_3[0]:.3f}, {coefficients_3[1]:.3f}), {f_3}")
        ax[i].plot(x, [funcs[i](j, coefficients_3[0], coefficients_3[1]) for j in x], color="r", label="nelder_mead")

        ax[i].legend()

    plt.subplots_adjust(0.05, 0.04, 0.95, 0.9)

    plt.suptitle("Direct methods")
    plt.show()
