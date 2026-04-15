import networkx as nx

def build_graph():
    G = nx.DiGraph()
    G.add_nodes_from(range(10))
    src, mon = 0, 9
    edge_data = [
        (0, 1, 0.1), (1, 2, 0.1), (2, 3, 0.1), (3, 4, 0.1),
        (4, 5, 10.0), (5, 6, 10.0), (6, 7, 10.0),
        (0, 8, 0.5), (8, 9, 0.5)
    ]
    for u, v, r in edge_data:
        G.add_edge(u, v, rate=r)
    
    return G, src, mon