import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.linalg import solve
import os
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# 1. CORE MATH: SHS & FRESHNESS
# ═══════════════════════════════════════════════════════════════════════════

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
    return G, src, mon

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

def freshness(G, active_edges, src=0):
    if not active_edges: return 1e-5
    return 1.0 / np.mean(solve_shs(G, active_edges, src))

# ═══════════════════════════════════════════════════════════════════════════
# 2. DIVERGENT GREEDY ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════

def matroidal_greedy_modified(G, k, src=0, weight=0.2):
    """Matroid: Picks any k edges. Tempted by high rates anywhere."""
    sel, rem, fh, gh = [], list(G.edges()), [], []
    for _ in range(k):
        fc = freshness(G, sel, src)
        best_score, best_e = -1.0, None
        for e in rem:
            # Score = Actual Freshness Gain + Temptation (Local Rate)
            score = (freshness(G, sel + [e], src) - fc) + (weight * G.edges[e]['rate'])
            if score > best_score:
                best_score, best_e = score, e
        if best_e is None: break
        sel.append(best_e); rem.remove(best_e)
        fh.append(freshness(G, sel, src))
        gh.append(best_score)
    return sel, fh, gh

def greedoidal_greedy_proper(G, k, src=0, weight=0.2):
    """Greedoid: Only picks edges reachable from source."""
    sel, rem, fh, gh = [], list(G.edges()), [], []
    for _ in range(k):
        H = nx.DiGraph(); H.add_nodes_from(G.nodes()); H.add_edges_from(sel)
        reach = {src} | nx.descendants(H, src)
        frontier = [e for e in rem if e[0] in reach]
        if not frontier: break
        
        fc = freshness(G, sel, src)
        best_score, best_e = -1.0, None
        for e in frontier:
            score = (freshness(G, sel + [e], src) - fc) + (weight * G.edges[e]['rate'])
            if score > best_score:
                best_score, best_e = score, e
        if best_e is None: break
        sel.append(best_e); rem.remove(best_e)
        fh.append(freshness(G, sel, src))
        gh.append(best_score)
    return sel, fh, gh

# ═══════════════════════════════════════════════════════════════════════════
# 3. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

BG, GRID, DARK, GRAY = '#1a1d27', '#2a2d3a', '#0f1117', '#90a4ae'
BLUE, GREEN, ORANGE, RED, GOLD, PINK = '#4fc3f7', '#81c784', '#ffb74d', '#ef9a9a', '#ffd54f', '#f48fb1'

def sax(ax, title=''):
    ax.set_facecolor(BG)
    ax.tick_params(colors=GRAY, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, lw=0.6)
    if title: ax.set_title(title, color='white', fontsize=10, fontweight='bold')

def draw_topo(ax, G, pos, sel, src, mon, ec, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color='white', fontsize=9, pad=10)
    
    # Reachability check
    H = nx.DiGraph(); H.add_nodes_from(G.nodes()); H.add_edges_from(sel if sel else [])
    reachable = {src} | nx.descendants(H, src)
    
    # Background Edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GRAY, alpha=0.05, width=0.5)
    
    if sel:
        live = [e for e in sel if e[0] in reachable]
        dead = [e for e in sel if e[0] not in reachable]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=live, edge_color=ec, width=2.5, arrows=True)
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=dead, edge_color=RED, width=1.5, style='--', alpha=0.5)

    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[src], node_color=GOLD, node_size=500, node_shape='*')
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[mon], node_color=PINK, node_size=400, node_shape='D')
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[n for n in G.nodes() if n not in [src, mon]], node_color=GRAY, node_size=200, alpha=0.3)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color='white', font_size=7)
    ax.axis('off')

def run_and_plot(k=8):
    G, src, mon = build_graph()
    pos = nx.spring_layout(G, seed=42)
    
    m_e, m_fh, m_gh = matroidal_greedy_modified(G, k, src)
    g_e, g_fh, g_gh = greedoidal_greedy_proper(G, k, src)
    
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(DARK)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1])

    # Top Row: Topologies
    draw_topo(fig.add_subplot(gs[0, 0]), G, pos, m_e, src, mon, BLUE, "Matroidal (Disconnected Allowed)")
    draw_topo(fig.add_subplot(gs[0, 1]), G, pos, g_e, src, mon, GREEN, "Greedoidal (Strictly Reachable)")

    # Bottom Left: Freshness Curve
    ax_f = fig.add_subplot(gs[1, 0])
    sax(ax_f, "Network Freshness f(S) vs Links Added")
    ax_f.plot(range(1, k+1), m_fh, 'o-', color=BLUE, label='Matroid')
    ax_f.plot(range(1, k+1), g_fh, 's-', color=GREEN, label='Greedoid')
    ax_f.legend()

    # Bottom Right: Marginal Gains
    ax_g = fig.add_subplot(gs[1, 1])
    sax(ax_g, "Marginal Utility (Score) per Step")
    w = 0.35
    ax_g.bar(np.arange(1, k+1)-w/2, m_gh, width=w, color=BLUE, alpha=0.6, label='Matroid Score')
    ax_g.bar(np.arange(1, k+1)+w/2, g_gh, width=w, color=GREEN, alpha=0.6, label='Greedoid Score')
    ax_g.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_and_plot(k=9)