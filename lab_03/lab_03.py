from random import random, normalvariate, seed


def get_signal() -> list:
    alpha, beta = random(), random()
    delta_lst = [normalvariate(0, 1) for i in range(101)]

    y_lst = [alpha * k / 100. + beta + delta_lst[k] for k in range(101)]

    return y_lst


if __name__ == '__main__':
    seed()
    print(get_signal())
