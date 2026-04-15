import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from freshness import universal_freshness
from matroidal_chosing import matroidal
from greedoidal_chosing import greedoidal

# Parameters
num_graphs = 10
N, K, src = 20, 25, 0  # K > N means you are looking for a dense sub-topology
m_scores, g_scores = [], []

print(f"Starting simulation on {num_graphs} graphs...")

for i in range(num_graphs):
    # Generate random directed graph
    G = nx.fast_gnp_random_graph(N, 0.4, directed=True)
    
    # Guarantee source connectivity so the experiment is valid
    if G.out_degree(src) == 0:
        target = np.random.randint(1, N)
        G.add_edge(src, target)
    
    for (u, v) in G.edges():
        G.edges[u, v]['rate'] = np.random.uniform(1.0, 10.0)

    # 1. Run Policies
    # Capturing (sel, fh, gh) from your functions
    res_m = matroidal(G, K, src)
    res_g = greedoidal(G, K, src)
    
    # Extract the last value of the freshness history (fh)
    fh_m = res_m[1] 
    fh_g = res_g[1]

    # Store raw scores since we can't calculate f_star (Optimal) for these dimensions
    m_scores.append(fh_m[-1] if fh_m else 0)
    g_scores.append(fh_g[-1] if fh_g else 0)

# --- Visualization ---

plt.figure(figsize=(12, 6))

# Plot 1: Distribution of Final Freshness
plt.subplot(1, 2, 1)
plt.hist(m_scores, bins=20, alpha=0.5, label='Matroidal', color='blue')
plt.hist(g_scores, bins=20, alpha=0.5, label='Greedoidal', color='green')
plt.title(f'Freshness Distribution (N={N}, K={K})')
plt.xlabel('Total Freshness $\hat{f}(S)$')
plt.ylabel('Frequency')
plt.legend()

# Plot 2: Direct Comparison (Scatter)
plt.subplot(1, 2, 2)
plt.scatter(m_scores, g_scores, alpha=0.6, c='purple')
# Line of equality
lims = [0, max(max(m_scores), max(g_scores))]
plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Equal Performance')
plt.title('Matroidal vs Greedoidal')
plt.xlabel('Matroidal Final Score')
plt.ylabel('Greedoidal Final Score')
plt.legend()

plt.tight_layout()
plt.show()

# Print the average gap
avg_gap = np.mean(np.array(m_scores) - np.array(g_scores))
print(f"Average Freshness Gap: {avg_gap:.4f}")