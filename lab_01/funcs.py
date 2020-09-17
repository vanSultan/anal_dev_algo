from functools import reduce


def constant_function(v: list, c: float = 1.) -> float:
    return c


def sum_function(v: list) -> float:
    return sum(v)


def prod_function(v: list) -> float:
    return reduce(lambda a, b: a * b, v)


def naive_poly_function(v: list, x: float = 1.5) -> float:
    return sum([val * pow(x, k) for k, val in enumerate(v)])


def gorner_method_function(v: list, x: float = 1.5) -> float:
    def gorner_sub(lst: list, value: float, aggregate: float) -> float:
        if len(lst) == 0:
            return aggregate
        else:
            agg = lst[-1] + value * aggregate
            return gorner_sub(lst[:-1], value, agg)

    return gorner_sub(v, x, 1)


if __name__ == '__main__':
    print(gorner_method_function([1, 2, 3]))
