from random import seed, randint

import matplotlib.pyplot as plt
import networkx as nx

seed()


def a_star():
    a_graph = nx.grid_graph(dim=(10, 10))
    pos = dict((n, n) for n in a_graph.nodes())
    for j in range(4, 10):
        for i in range(1, 5):
            a_graph.remove_node((i, j))
        a_graph.remove_node((j, 1))
    start_node, end_node = (8, 0), (9, 9)
    print(start_node, end_node)
    a_star_path = nx.algorithms.astar_path(a_graph, start_node, end_node)
    print(a_star_path)

    c_map = []
    for node in a_graph.nodes:
        if node in a_star_path:
            c_map.append('red')
        else:
            c_map.append('blue')

    nx.draw(a_graph, pos, with_labels=True, font_size=8, node_size=500, node_color=c_map)
    plt.show()


if __name__ == '__main__':
    g = nx.gnm_random_graph(100, 500)
    for pair in g.edges:
        g[pair[0]][pair[1]]['weight'] = randint(1, 100)

    p_1 = nx.algorithms.dijkstra_path(g, 0, 99)
    p_2 = nx.algorithms.bellman_ford_path(g, 0, 99)

    print(p_1)
    print(p_2)
