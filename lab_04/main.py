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

    # print(titles[i])
    # co_1, count = gradient_descent(f_least_squares, [0.1, 0.1], (funcs[i], signal), precision=eps, iterations=200)
    # print("gradient_descent ((a, b), iter, f_c, df_c): ({co[0]:.3f}, {co[1]:.3f}), {it}, {f_c}, {d_c}".format(
    #     co=co_1, it=count, f_c=count, d_c=count
    # ))
    # ax[i].plot(x, [funcs[i](j, co_1[0], co_1[1]) for j in x], color="c", label="gradient_descent")
    #
    # co_2 = minimize(f_least_squares, array([0.2, 0.9]), (funcs[i], signal), method="CG", tol=eps)
    # print(
    #     "conjugate_gradient_descent ((a, b), iter, f_c, df_c): "
    #     "({c[0]:.3f}, {c[1]:.3f}), {it}, {f_c}, {d_c}".format(c=co_2.x, it=co_2.nit, f_c=co_2.nfev, d_c=co_2.njev)
    # )
    # ax[i].plot(x, [funcs[i](j, co_2.x[0], co_2.x[1]) for j in x], color="m", label="conjugate_gradient_descent")
    #
    # co_3 = minimize(f_least_squares, array([0.3, -0.5]), (linear_approx, signal), 'Newton-CG',
    #                 jac=lambda xk, *args: approx_fprime(xk, f_least_squares, eps, *args), tol=eps)
    # print(
    #     "newton ((a, b), iter, f_c, df_c): "
    #     "({c[0]:.3f}, {c[1]:.3f}), {it}, {f_c}, {d_c}".format(c=co_2.x, it=co_2.nit, f_c=co_2.nfev, d_c=co_2.njev)
    # )
    # ax[i].plot(x, [funcs[i](j, co_3.x[0], co_3.x[1]) for j in x], color="r", label="newton")
    #
    # co_4 = least_squares(f_least_squares, array([0.6, 0.2]), args=(funcs[i], signal))
    # print(
    #     "levenberg_marquardt ((a, b), iter, f_c, df_c): "
    #     "({c[0]:.3f}, {c[1]:.3f}), {it}, {f_c}, {d_c}".format(c=co_2.x, it=co_2.nit, f_c=co_2.nfev, d_c=co_2.njev)
    # )
    # ax[i].plot(x, [funcs[i](j, co_4.x[0], co_4.x[1]) for j in x], color="y", label="levenberg_marquardt")

    plt.legend()
    # plt.subplots_adjust(0.05, 0.04, 0.95, 0.9)

    plt.suptitle("Stochastic and metaheuristic algorithms")
    plt.show()
