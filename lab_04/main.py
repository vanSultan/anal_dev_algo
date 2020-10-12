from random import seed
from typing import List

from matplotlib import pyplot as plt
from numpy import array, NaN

from lab_04 import *


def print_log(func_name: str, coefficient: List[float], iterations_count: int, fun_res: float) -> None:
    print(
        "{f} ((a, b, c, d), iter, f_c): ({co[0]:.3f}, {co[1]:.3f}, {co[2]:.3f}, {co[3]:.3f}), {it}, {f_c}".format(
            f=func_name, co=coefficient, it=iterations_count, f_c=fun_res
        )
    )


if __name__ == '__main__':
    seed()
    eps = 1e-3
    signal = get_signal()

    x, y = list(map(lambda k: k[0], signal)), list(map(lambda k: k[1], signal))
    limits = ((min(x), min(y)), (max(x), max(y)))

    title = "Rational approximation"

    plt.title(title)
    plt.grid(1)
    plt.scatter(x, y, color="blue", marker=".", label="signal")

    co_1 = minimize(f_least_squares, array([0.1, 0.1, 0.1, 0.1]), (rational_approx, signal), "Nelder-Mead")
    print_log("Nelder-Mead", co_1.x, co_1.nit, co_1.nfev)
    plt.plot(x, [rational_approx(x_i, *co_1.x) for x_i in x], color="m", label="Nelder-Mead")

    co_2 = least_squares(f_least_squares, array([-0.1, 0.1, -0.1, 0.1]), args=(rational_approx, signal))
    print_log("Levenberg-Marquardt", co_2.x, NaN, co_2.nfev)
    plt.plot(x, [rational_approx(x_i, *co_2.x) for x_i in x], color="r", label="Levenberg-Marquardt")

    co_3 = dual_annealing(f_least_squares, [(-5, 5), (-5, 5), (-5, 5), (-5, 5)], (rational_approx, signal))
    print_log("Simulated Annealing", co_3.x, co_3.nit, co_3.nfev)
    plt.plot(x, [rational_approx(x_i, *co_3.x) for x_i in x], color="y", label="Simulated Annealing")

    co_4 = differential_evolution(f_least_squares, [(-5, 5), (-5, 5), (-5, 5), (-5, 5)], (rational_approx, signal))
    print_log("Differential Evolution", co_4.x, co_4.nit, co_4.nfev)
    plt.plot(x, [rational_approx(x_i, *co_4.x) for x_i in x], color="g", label="Differential Evolution")

    plt.legend()

    plt.suptitle("Stochastic and metaheuristic algorithms")
    plt.show()
