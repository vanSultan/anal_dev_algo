import os
from random import random, seed
from timeit import timeit
from typing import List

import pandas as pd

import lab_01


def get_random_matrix(n: int, m: int) -> List[list]:
    return [[random() for j in range(m)] for i in range(n)]


def get_random_vector(n: int) -> list:
    return [random() for i in range(n)]


def matrix_copy(matr: List[list]) -> List[list]:
    return [row.copy() for row in matr]


if __name__ == '__main__':
    seed()
    df = pd.DataFrame(columns=lab_01.__all__, dtype='float')
    n = 100
    step = 10

    skip_function = set()

    if not os.path.isdir("dumps"):
        os.mkdir("dumps")

    for i in range(1, n + 1):
        vector = get_random_vector(i)
        matrix_a = get_random_matrix(i, i)
        matrix_b = get_random_matrix(i, i)

        row = dict()
        for foo in filter(lambda x: x not in skip_function, lab_01.__all__):
            time_s = None
            try:
                time_s = timeit(f"{foo}(vector)", setup=f"from lab_01 import {foo}", number=5,
                                globals=dict(vector=vector.copy()))
            except (TypeError, ValueError):
                time_s = timeit(f"{foo}(matrix_a, matrix_b)", setup=f"from lab_01 import {foo}", number=5,
                                globals=dict(matrix_a=[r.copy() for r in matrix_a],
                                             matrix_b=[r.copy() for r in matrix_b]))
            except OverflowError:
                skip_function.add(foo)
            row[foo] = time_s

        for foo in skip_function:
            row[foo] = None

        df.loc[i] = row

        if i % step == 0:
            print(f"save dump: n = {i}")
            df.to_csv(f"dumps/{i}_dump.csv", sep=";")

    print(df[df.columns[-3:]])
