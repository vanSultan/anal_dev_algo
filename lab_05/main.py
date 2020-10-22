import networkx as nx
import matplotlib.pyplot as plt


if __name__ == '__main__':
    g = nx.gnm_random_graph(100, 200)

    print("Adjacency list:")
    for i, node in enumerate(g.nodes):
        if i > 10: break
        print(f"{node} {set(g.adj[node])}")
    print()

    print("Adjacency matrix:")
    print(nx.adjacency_matrix(g).toarray())

    nx.draw_shell(g, **{
        "node_size": 50,
        "node_shape": "8",
        "width": 0.7
    })
    plt.show()
