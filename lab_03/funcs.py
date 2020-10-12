from random import random, normalvariate
from typing import List, Tuple, Callable, Iterable

from numpy import finfo, ndarray, array
from scipy.optimize import approx_fprime


def linear_approx(x: float, a: float, b: float) -> float:
    return a * x + b


def rational_approx(x: float, a: float, b: float) -> float:
    return a / (1 + b * x)


def f_least_squares(theta: Tuple[float], foo: Callable, signal: List[Tuple[float, float]]) -> float:
    d = sum(pow(foo(x_k, *theta) - y_k, 2) for x_k, y_k in signal)
    return d


def get_signal(n: int = 101) -> List[Tuple[float, float]]:
    alpha, beta = random(), random()
    signal = [(x, alpha * x + beta + delta) for x, delta in [(k / 100., normalvariate(0, 1)) for k in range(n)]]

    return signal


def gradient_descent(foo: Callable, x0: Iterable, args: Tuple, m: int = None, alpha: float = 1e-3,
                     precision: float = finfo(float).eps, iterations: int = 10000) -> Tuple[List[float], int]:
    vec_x = list(x0)
    iteration = 1
    if not m:
        m = len(vec_x)

    while iteration <= iterations:
        foo_cur = foo(vec_x, *args)
        gradient = approx_fprime(vec_x, foo, alpha, *args)

        for i in range(m):
            vec_x[i] += -alpha * gradient[i]

        if abs(foo_cur) <= precision:
            break

        iteration += 1

    return vec_x, iteration

if __name__ == '__main__':
    sig = get_signal()
    ax, ay = zip(*sig)
    from scipy import optimize
    res = optimize.minimize(f_least_squares, array([0.1, 0.1]), (linear_approx, sig), 'CG', tol=1e-3)
    print(res.x)
    print(gradient_descent(f_least_squares, [0.1, 0.1], (linear_approx, sig), precision=1e-3, iterations=1000)[0])
    res = optimize.minimize(f_least_squares, array([0.1, 0.1]), (linear_approx, sig), 'Newton-CG', jac=lambda xk, *args: approx_fprime(xk, f_least_squares, 1e-3, *args), tol=1e-3)
    print(res.x)
    # res = optimize.least_squares(lambda theta, x: theta[0] * x + theta[1], [0.1, 0.1], method='lm', ftol=1e-3, args=(array(ax), ))
    # print(res.x)
    res = optimize.least_squares(f_least_squares, [0.1, 0.1], args=(linear_approx, sig))
    print(res.x)
