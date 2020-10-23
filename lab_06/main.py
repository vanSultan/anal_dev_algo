from random import seed, choice
from timeit import timeit
from typing import Callable, List

import matplotlib.pyplot as plt
import networkx as nx

seed()
graph_type = nx.classes.graph.Graph


def a_star():
    graph = nx.grid_graph(dim=(10, 10))
    pos = dict((n, n) for n in graph.nodes())
    for j in range(4, 10):
        for i in range(1, 5):
            graph.remove_node((i, j))
        graph.remove_node((j, 1))

    draw_common_options = dict(
        G=graph, pos=pos, with_labels=True, font_size=8, node_size=500
    )

    fig, ax = plt.subplots(2, 3)
    fig.canvas.set_window_title("A* algorithm")
    fig.suptitle("A* algorithm")
    fig.subplots_adjust(0.01, 0.02, 0.99, 0.90)

    nx.draw(**draw_common_options, ax=ax[0, 0])
    ax[0, 0].set_title("Graph")

    points = list()
    for i in range(2):
        for j in range(3):
            if i == 0 and j == 0:
                continue
            while True:
                start_node, end_node = choice(list(graph.nodes)), choice(list(graph.nodes))
                if (start_node, end_node) not in points and start_node != end_node:
                    points.append((start_node, end_node))
                    break
            a_star_path = nx.algorithms.astar_path(graph, start_node, end_node)

            c_map = list()
            for node in graph.nodes:
                if node in a_star_path:
                    c_map.append('red')
                else:
                    c_map.append('blue')

            nx.draw(**draw_common_options, ax=ax[i, j], node_color=c_map)
            ax[i, j].set_title(f"s = {start_node}, t = {end_node}")


def routine_sp(graph: graph_type, source: int, method: Callable[..., List]) -> float:
    nodes = list(graph.nodes)
    nodes.remove(source)

    time_s = timeit(f"[f(g, s, n) for n in nodes]", number=10, globals=dict(f=method, g=graph, s=source, nodes=nodes))

    return time_s


def shortest_path():
    graph = nx.gnm_random_graph(100, 500)
    start_node = choice(list(graph.nodes))
    dij_time = routine_sp(graph, start_node, nx.algorithms.dijkstra_path)
    bf_time = routine_sp(graph, start_node, nx.algorithms.bellman_ford_path)
    print(f"Dijkstra: {round(dij_time, 3)} | Bellman-Ford: {round(bf_time, 3)} | div: {round(bf_time / dij_time, 3)}")


if __name__ == '__main__':
    shortest_path()
    a_star()
    plt.show()
