import numpy as np
def universal_freshness(G, S, source=0):
    """
    G: NetworkX DiGraph
    S: List of chosen edges (u, v)
    source: The node with age 0
    """
    n = G.number_of_nodes()
    A = np.zeros((n, n))
    b = -np.ones(n) # The '-1' represents the linear aging over time

    for (u, v) in S:
        rate = G.edges[u, v]['rate']
        A[v, u] += rate   # Flow of fresh information from u to v
        A[v, v] -= rate   # Rate of departure from old state at v

    # Boundary condition: Source age is always 0
    A[source, :] = 0
    A[source, source] = -1
    b[source] = 0

    # Solve the system
    try:
        expected_ages = np.linalg.solve(A, b)
        # Ensure we don't have negative age due to numerical instability
        expected_ages = np.maximum(expected_ages, 0) 
        return 1.0 / np.mean(expected_ages)
    except np.linalg.LinAlgError:
        return 0.0 # Network is disconnected