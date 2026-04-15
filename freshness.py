import numpy as np
from scipy.linalg import solve
import networkx as nx

def solve_shs(G, active_edges, src=0):
    n = G.number_of_nodes()
    A = np.zeros((n, n))
    b = -np.ones(n)
    for (u, v) in active_edges:
        r = G.edges[u, v]['rate']
        A[v, u] += r
        A[v, v] -= r
    A[src, :] = 0.0
    A[src, src] = -1.0
    b[src] = 0.0
    for i in range(n):
        if A[i, i] == 0.0: A[i, i] = -1e-5
    try:
        return np.maximum(solve(A, b), 0.01)
    except:
        return np.ones(n) * 500.0

def universal_freshness(G, active_edges, src):
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes()) 
    H.add_edges_from(active_edges)
    reachable_nodes = {src} | nx.descendants(H, src)
    ages = solve_shs(G, active_edges, src)
    reachable_ages = [ages[i] for i in reachable_nodes]
    total_freshness = sum(1.0 / age for age in reachable_ages)
    return total_freshness