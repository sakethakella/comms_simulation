import networkx as nx

def build_graph():
    G = nx.DiGraph()
    # A more complex graph with 25 nodes
    G.add_nodes_from(range(25))
    src, mon = 0, 24
    
    # Adding multiple paths, diverse rates, bottlenecks, and cross-links
    edge_data = [
        # Main Path 1 (Slow but reliable)
        (0, 1, 0.1), (1, 2, 0.1), (2, 3, 0.1), (3, 4, 0.1), (4, 5, 0.1),
        (5, 6, 0.1), (6, 7, 0.1), (7, 8, 0.1), (8, 9, 0.1), (9, 24, 0.1),
        
        # Main Path 2 (Fast but with a severe bottleneck in the middle)
        (0, 10, 5.0), (10, 11, 5.0), (11, 12, 0.005), (12, 13, 5.0), (13, 24, 5.0),
        
        # Main Path 3 (Moderate speed, multiple hops)
        (0, 14, 1.0), (14, 15, 1.0), (15, 16, 1.5), (16, 17, 1.0), (17, 18, 1.0), (18, 24, 1.0),
        
        # Main Path 4 (High variance)
        (0, 19, 10.0), (19, 20, 0.5), (20, 21, 20.0), (21, 22, 0.2), (22, 23, 10.0), (23, 24, 0.5),
        
        # Cross Connectors
        (2, 11, 0.5), (11, 15, 2.0), (16, 21, 3.0), (21, 6, 1.0),
        (4, 16, 0.8), (17, 22, 4.0), (12, 7, 0.1), (8, 23, 5.0),
        (1, 10, 0.2), (14, 19, 1.2), (3, 20, 6000), (13, 18, 100.0),
        
        # Loops and Back-edges
        (5, 2, 0.05), (15, 14, 0.1), (22, 19, 0.1),
        
        # Direct Express Links
        (0, 24, 0.01),   # Direct but very slow
        (10, 17, 2.5),
        (16, 24, 0.8)
    ]
    
    for u, v, r in edge_data:
        G.add_edge(u, v, rate=r)
    
    return G, src, mon