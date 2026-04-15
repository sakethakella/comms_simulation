import networkx as nx
import matplotlib.pyplot as plt

def build_graph():
    G = nx.DiGraph()
    G.add_nodes_from(range(10))
    src, mon = 0, 9
    # Topology with a "trap": high rate links (5.0) are deep in the graph
    edge_data = [
        (0, 1, 0.5), (1, 2, 0.8), (2, 3, 0.8), (3, 9, 0.8),
        (0, 4, 0.3), (4, 5, 5.0), (5, 6, 5.0), (6, 9, 5.0),
        (1, 5, 2.5), (2, 6, 2.8), (3, 6, 2.0), (4, 6, 3.5),
        (0, 7, 0.4), (7, 8, 4.0), (8, 9, 4.0),
    ]
    for u, v, r in edge_data:
        G.add_edge(u, v, rate=r)
    nx.draw(G,with_labels=True, node_color='lightblue', font_weight='bold')
    plt.show()
    return G, src, mon