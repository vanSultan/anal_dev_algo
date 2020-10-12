from random import seed, random

from matplotlib import pyplot as plt
from numpy import array
from scipy.optimize import approx_fprime

from lab_03 import *

if __name__ == '__main__':
    seed()
    eps = 1e-3
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
        co_1, count = gradient_descent(f_least_squares, [0.1, 0.1], (funcs[i], signal), precision=eps, iterations=200)
        print("gradient_descent ((a, b), iter, f_c, df_c): ({co[0]:.3f}, {co[1]:.3f}), {it}, {f_c}, {d_c}".format(
            co=co_1, it=count, f_c=count, d_c=count
        ))
        ax[i].plot(x, [funcs[i](j, co_1[0], co_1[1]) for j in x], color="c", label="gradient_descent")

        co_2 = minimize(f_least_squares, array([0.2, 0.9]), (funcs[i], signal), method="CG", tol=eps)
        print(
            "conjugate_gradient_descent ((a, b), iter, f_c, df_c): "
            "({c[0]:.3f}, {c[1]:.3f}), {it}, {f_c}, {d_c}".format(c=co_2.x, it=co_2.nit, f_c=co_2.nfev, d_c=co_2.njev)
        )
        ax[i].plot(x, [funcs[i](j, co_2.x[0], co_2.x[1]) for j in x], color="m", label="conjugate_gradient_descent")

        co_3 = minimize(f_least_squares, array([0.3, -0.5]), (linear_approx, signal), 'Newton-CG',
                        jac=lambda xk, *args: approx_fprime(xk, f_least_squares, eps, *args), tol=eps)
        print(
            "newton ((a, b), iter, f_c, df_c): "
            "({c[0]:.3f}, {c[1]:.3f}), {it}, {f_c}, {d_c}".format(c=co_2.x, it=co_2.nit, f_c=co_2.nfev, d_c=co_2.njev)
        )
        ax[i].plot(x, [funcs[i](j, co_3.x[0], co_3.x[1]) for j in x], color="r", label="newton")

        co_4 = least_squares(f_least_squares, array([0.6, 0.2]), args=(funcs[i], signal))
        print(
            "levenberg_marquardt ((a, b), iter, f_c, df_c): "
            "({c[0]:.3f}, {c[1]:.3f}), {it}, {f_c}, {d_c}".format(c=co_2.x, it=co_2.nit, f_c=co_2.nfev, d_c=co_2.njev)
        )
        ax[i].plot(x, [funcs[i](j, co_4.x[0], co_4.x[1]) for j in x], color="y", label="levenberg_marquardt")

        ax[i].legend()

    plt.subplots_adjust(0.05, 0.04, 0.95, 0.9)

    plt.suptitle("First- and second-order methods")
    plt.show()
