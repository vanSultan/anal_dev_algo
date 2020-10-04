import argparse
import os
from datetime import datetime
from functools import reduce
from random import random, seed
from timeit import timeit
from typing import List

import pandas as pd
from matplotlib import pyplot as plt
from numpy import polyfit

import lab_01

seed()


def poly(coefficients: list, x: float) -> float:
    return reduce(lambda agg, a: agg + a[1] * pow(x, a[0]), enumerate(reversed(coefficients)), 0)
  

def get_argparser():
    parser = argparse.ArgumentParser(prog="lab_01")
    parser.add_argument("-c", "--csv_path", type=str, help="Path to csv of dataframe")
    parser.add_argument("-s", "--skip_methods", nargs="*", type=str, help="List of ignored methods",
                        choices=lab_01.__all__, default=["prod_matrix", "vinograd_mod"])
    parser.add_argument("-n", type=int, help="Max vector size", default=2000)
    parser.add_argument("-sd", "--step_dump", type=int, help="Step of dump", default=200)
    parser.add_argument("-sr", "--step_run", type=int, help="Step of run", default=1)
    return parser.parse_args()


def get_random_matrix(n: int, m: int) -> List[list]:
    return [[random() for j in range(m)] for i in range(n)]


def get_random_vector(n: int) -> list:
    return [random() for i in range(n)]


def matrix_copy(matr: List[list]) -> List[list]:
    return [row.copy() for row in matr]


if __name__ == '__main__':
    args = get_argparser()

    n = args.n
    step_run = args.step_run
    if args.csv_path:
        df = pd.read_csv(args.csv_path, sep=';', header=0, index_col=0)
    else:
        function_set = set(lab_01.__all__) - set(args.skip_methods)
        skip_func = list()
        df = pd.DataFrame(columns=function_set, dtype='float')

        step_dump = args.step_dump

        if not os.path.isdir("dumps"):
            os.mkdir("dumps")

        for i in range(1, n + 1, step_run):
            vector = get_random_vector(i)
            matrix_a = get_random_matrix(i, i)
            matrix_b = get_random_matrix(i, i)

            row = dict()
            for foo in filter(lambda f: f not in skip_func, function_set):
                time_s = None
                try:
                    time_s = timeit(f"{foo}(vector)", setup=f"from lab_01 import {foo}", number=5,
                                    globals=dict(vector=vector.copy()))
                except (TypeError, ValueError):
                    time_s = timeit(f"{foo}(matrix_a, matrix_b)", setup=f"from lab_01 import {foo}", number=5,
                                    globals=dict(matrix_a=[r.copy() for r in matrix_a],
                                                 matrix_b=[r.copy() for r in matrix_b]))
                except OverflowError:
                    skip_func.append(foo)
                row[foo] = time_s

            for foo in set(lab_01.__all__) - function_set:
                row[foo] = None

            df.loc[i] = row

            if i % step_dump == 0:
                print(f"save dump: n = {i}")
                df.to_csv(f"dumps/{datetime.now().strftime('%Y%m%d')}_{i}.csv", sep=";")

        print(f"save dump: n = {n}")
        df.to_csv(f"dumps/{datetime.now().strftime('%Y%m%d')}_{n}.csv", sep=";")

    poly_power = {
        "constant_function": 0, "sum_function": 1, "prod_function": 1,
        "naive_poly_function": 1, "horner_method_function": 1,
        "bubble_sort": 2, "quick_sort": 1, "timsort": 1,
        "matmul": 2
    }

    df = df.fillna(method='ffill')
    for col_name in df.columns:
        cof = polyfit(df.index, df[col_name], poly_power[col_name])
        plt.plot(df.index, df[col_name], label="noisy signal", color="m")
        plt.plot(df.index, [poly(list(cof), x) for x in df.index], label="approximated line", color="c")
        plt.title(col_name)
        plt.xlabel("len of vector")
        plt.ylabel("time, s")
        plt.grid(True)
        plt.legend()
        plt.show()
