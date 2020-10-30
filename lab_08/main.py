from random import seed
from timeit import timeit

import networkx as nx

seed()
graph_type = nx.classes.graph.Graph


def routine_sp(graph: graph_type) -> float:
    time_s = timeit("f(g)", number=10, globals=dict(f=nx.floyd_warshall, g=graph))

    return time_s


if __name__ == '__main__':
    g = nx.gnm_random_graph(100, 500, 0, True)
    print(routine_sp(g))
