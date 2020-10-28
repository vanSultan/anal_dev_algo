from random import randint, seed

import networkx as nx
import matplotlib.pyplot as plt


seed()


if __name__ == '__main__':
    count_nodes = 100
    count_edges = 200

    g = nx.gnm_random_graph(count_nodes, count_edges)

    print("Adjacency list:")
    for i, node in enumerate(g.nodes):
        if i > 10: break
        print(f"{node} {set(g.adj[node])}")
    print()

    print("Adjacency matrix:")
    print(nx.adjacency_matrix(g).toarray())

    start_node = randint(0, count_nodes - 1)
    end_node = randint(0, count_nodes - 1)
    while end_node == start_node:
        end_node = randint(0, count_nodes - 1)
    print()

    dfs_edges = nx.dfs_edges(g, start_node)
    dfs_path = list()
    for pair in dfs_edges:
        dfs_path.append(pair)
        if pair[1] == end_node:
            break
    print(f"[DFS] Connected components: {dfs_path}")

    bfs_tree = nx.bfs_tree(g, start_node)
    k, v = end_node, list(bfs_tree.pred[end_node].keys())[0]
    bfs_path = [end_node]
    while v != start_node:
        bfs_path.append(v)
        k, v = v, list(bfs_tree.pred[v].keys())[0]
    bfs_path = [start_node] + list(reversed(bfs_path))
    print(f"[BFS] Shortest path between nodes {start_node} and {end_node}: {bfs_path}")
    print(f"[BFS] Len of path: {len(bfs_path)}")

    nx.draw_shell(g, **{
        "node_size": 50,
        "node_shape": "8",
        "width": 0.7
    })
    plt.show()
