"""
=============================================================================
  CTMC-DRIVEN ADAPTIVE LINK SELECTION WITH REAL-TIME FRESHNESS SCORING
=============================================================================

  HOW IT WORKS:
  ─────────────
  1. A CTMC (Gillespie) runs continuously as the "ground truth" dynamics.
  2. The MONITOR NODE acts as a sensor — whenever it detects a version-age
     spike (stale detection), it TRIGGERS a re-evaluation of the network.
  3. On trigger, the SHS moment equations are solved to compute current
     expected ages E[X] → freshness f(S) for every candidate link.
  4. The MATROIDAL or GREEDOIDAL greedy then selects the best k links
     based on that freshness score.
  5. The newly selected links are ACTIVATED in the CTMC going forward.
  6. We repeat: CTMC runs → sensor detects staleness → re-score →
     reselect links → CTMC resumes with new topology.

  This creates a closed feedback loop:
      CTMC dynamics → sensor trigger → SHS scoring → link selection → CTMC
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.linalg import solve
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# 1.  GRAPH
# ══════════════════════════════════════════════════════════════════════════

def build_graph():
    G = nx.DiGraph()
    G.add_nodes_from(range(10))
    src, mon = 0, 9
    edge_data = [
        (0, 1, 0.5), (1, 2, 0.8), (2, 3, 0.8), (3, 9, 0.8),
        (0, 4, 0.3), (4, 5, 5.0), (5, 6, 5.0), (6, 9, 5.0),
        (1, 5, 2.5), (2, 6, 2.8), (3, 6, 2.0), (4, 6, 3.5),
        (0, 7, 0.4), (7, 8, 4.0), (8, 9, 4.0),
        (1, 4, 1.0), (2, 5, 1.5), (3, 8, 1.2),
    ]
    for u, v, r in edge_data:
        G.add_edge(u, v, rate=r)
    return G, src, mon


# ══════════════════════════════════════════════════════════════════════════
# 2.  SHS MOMENT SOLVER  →  f(S)
# ══════════════════════════════════════════════════════════════════════════

def solve_shs(G, active_edges, src=0):
    """Solve A·E[X] = b (linearised SHS via lattice homomorphism)."""
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
        if A[i, i] == 0.0:
            A[i, i] = -1e-5
    try:
        return np.maximum(solve(A, b), 0.01)
    except Exception:
        return np.ones(n) * 500.0


def freshness(G, active_edges, src=0):
    """f(S) = 1 / mean(E[X])  — submodular objective."""
    if not active_edges:
        return 1e-5
    return 1.0 / np.mean(solve_shs(G, active_edges, src))


# ══════════════════════════════════════════════════════════════════════════
# 3.  LINK SELECTION ALGORITHMS  (triggered by sensor)
# ══════════════════════════════════════════════════════════════════════════

def matroidal_select(G, k, src=0):
    """
    Matroidal Greedy: pick any k edges maximising marginal freshness gain.
    Guarantee: f(S) >= (1-1/e)*f(OPT).
    """
    sel, rem = [], list(G.edges())
    for _ in range(k):
        fc = freshness(G, sel, src)
        best_g, best_e = -1.0, None
        for e in rem:
            g = freshness(G, sel + [e], src) - fc
            if g > best_g:
                best_g, best_e = g, e
        if best_e is None or best_g <= 1e-10:
            break
        sel.append(best_e)
        rem.remove(best_e)
    return sel


def greedoidal_select(G, k, src=0):
    """
    Greedoidal Greedy: pick edges only from the reachable frontier.
    F(S) = {(u,v) : u reachable from src in S}.
    Guarantee: f(S) >= f(OPT)/(1+curvature).
    """
    sel, rem = [], list(G.edges())
    for _ in range(k):
        if sel:
            H = nx.DiGraph()
            H.add_nodes_from(G.nodes())
            H.add_edges_from(sel)
            reach = {src} | nx.descendants(H, src)
        else:
            reach = {src}
        frontier = [e for e in rem if e[0] in reach]
        if not frontier:
            break
        fc = freshness(G, sel, src)
        best_g, best_e = -1.0, None
        for e in frontier:
            g = freshness(G, sel + [e], src) - fc
            if g > best_g:
                best_g, best_e = g, e
        if best_e is None or best_g <= 1e-10:
            break
        sel.append(best_e)
        rem.remove(best_e)
    return sel


# ══════════════════════════════════════════════════════════════════════════
# 4.  CTMC ENGINE WITH SENSOR TRIGGER
# ══════════════════════════════════════════════════════════════════════════

def run_ctmc_adaptive(G, src, mon, k,
                      algo='matroid',
                      src_rate=1.5,
                      stale_threshold=5,
                      reselect_cooldown=8.0,
                      T_max=300.0,
                      seed=42):
    """
    CTMC simulation with sensor-triggered link reselection.

    Parameters
    ----------
    stale_threshold  : version-age value that triggers a sensor alarm
    reselect_cooldown: minimum time between reselections (prevent thrashing)
    algo             : 'matroid' or 'greedoid'

    Returns
    -------
    dict with full trace data for plotting
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()

    # ── Initial link selection ──────────────────────────────────────────
    selector = matroidal_select if algo == 'matroid' else greedoidal_select
    active = selector(G, k, src)
    f_init = freshness(G, active, src)

    # ── State ───────────────────────────────────────────────────────────
    version = np.zeros(n, dtype=int)
    t = 0.0

    # ── Records ─────────────────────────────────────────────────────────
    times          = [0.0]
    ages           = [0]
    freshness_log  = [f_init]      # freshness score at each reselection
    trigger_times  = []            # when sensor fired
    trigger_ages   = []            # version age at trigger
    active_log     = [list(active)]# link sets over time
    reselect_times = [0.0]         # wall-clock times of reselections
    reselect_count = 0
    last_reselect  = -reselect_cooldown   # allow immediate first trigger

    rm = {(u, v): G.edges[u, v]['rate'] for (u, v) in active}

    while t < T_max:
        # Build event list
        evts = [('s',)]
        rats = [src_rate]
        for (u, v), r in rm.items():
            if version[u] > version[v]:
                evts.append(('c', u, v))
                rats.append(r)

        total = sum(rats)
        if total == 0:
            break

        dt = rng.exponential(1.0 / total)
        t += dt

        # Apply event
        idx = rng.choice(len(evts), p=np.array(rats) / total)
        ev = evts[idx]
        if ev[0] == 's':
            version[src] += 1
        else:
            version[ev[2]] = version[ev[1]]

        current_age = max(0, version[src] - version[mon])
        times.append(t)
        ages.append(current_age)

        # ── SENSOR DETECTION ─────────────────────────────────────────
        # Monitor node detects staleness when age exceeds threshold
        if (current_age >= stale_threshold and
                t - last_reselect >= reselect_cooldown):

            trigger_times.append(t)
            trigger_ages.append(current_age)

            # Reselect links based on CURRENT SHS scoring
            new_active = selector(G, k, src)
            f_new = freshness(G, new_active, src)

            active = new_active
            rm = {(u, v): G.edges[u, v]['rate'] for (u, v) in active}
            freshness_log.append(f_new)
            active_log.append(list(active))
            reselect_times.append(t)
            last_reselect = t
            reselect_count += 1

    print(f"  [{algo.upper():8s}] reselections={reselect_count:3d}  "
          f"final_f={freshness_log[-1]:.4f}  "
          f"mean_age={np.mean(ages[len(ages)//2:]):.2f}")

    return dict(
        times=np.array(times),
        ages=np.array(ages),
        freshness_log=freshness_log,
        trigger_times=trigger_times,
        trigger_ages=trigger_ages,
        active_log=active_log,
        reselect_times=reselect_times,
        final_active=active,
        reselect_count=reselect_count,
    )


# Static (no reselection) baseline
def run_ctmc_static(G, src, mon, active, src_rate=1.5, T_max=300.0, seed=42):
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    version = np.zeros(n, dtype=int)
    t = 0.0
    times, ages = [0.0], [0]
    rm = {(u, v): G.edges[u, v]['rate'] for (u, v) in active}
    while t < T_max:
        evts = [('s',)]; rats = [src_rate]
        for (u, v), r in rm.items():
            if version[u] > version[v]:
                evts.append(('c', u, v)); rats.append(r)
        total = sum(rats)
        if total == 0: break
        t += rng.exponential(1.0 / total)
        idx = rng.choice(len(evts), p=np.array(rats) / total)
        ev = evts[idx]
        if ev[0] == 's': version[src] += 1
        else: version[ev[2]] = version[ev[1]]
        ages.append(max(0, version[src] - version[mon]))
        times.append(t)
    return np.array(times), np.array(ages)


# ══════════════════════════════════════════════════════════════════════════
# 5.  ROLLING MEAN HELPER
# ══════════════════════════════════════════════════════════════════════════

def rmean(arr, w=250):
    if len(arr) < w:
        return np.arange(len(arr)), np.array(arr, dtype=float)
    ys = np.convolve(arr, np.ones(w) / w, mode='valid')
    return np.arange(len(ys)), ys


# ══════════════════════════════════════════════════════════════════════════
# 6.  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════

DARK   = '#0f1117'
BG     = '#1a1d27'
GRID   = '#2a2d3a'
GRAY   = '#90a4ae'
BLUE   = '#4fc3f7'
GREEN  = '#81c784'
ORANGE = '#ffb74d'
PINK   = '#f48fb1'
GOLD   = '#ffd54f'
RED    = '#ef9a9a'
PURPLE = '#ce93d8'


def sax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG)
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.xaxis.label.set_color(GRAY)
    ax.yaxis.label.set_color(GRAY)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    if title:
        ax.set_title(title, color='white', fontsize=9.5, fontweight='bold', pad=7)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def draw_topo(ax, G, pos, sel, src, mon, ec, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color='white', fontsize=9, fontweight='bold', pad=6)
    others = [v for v in G.nodes() if v not in [src, mon]]
    # Background edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GRAY, alpha=0.10,
                           arrows=True, arrowsize=8, width=0.5,
                           connectionstyle='arc3,rad=0.15')
    # Active selected edges
    if sel:
        H = nx.DiGraph(); H.add_nodes_from(G.nodes()); H.add_edges_from(sel)
        reach = {src} | nx.descendants(H, src)
        live  = [e for e in sel if e[0] in reach]
        dead  = [e for e in sel if e[0] not in reach]
        if live:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=live,
                                   edge_color=ec, alpha=0.92,
                                   arrows=True, arrowsize=14, width=2.5,
                                   connectionstyle='arc3,rad=0.15')
        if dead:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=dead,
                                   edge_color=RED, alpha=0.5,
                                   arrows=True, arrowsize=10, width=1.5,
                                   style='dashed',
                                   connectionstyle='arc3,rad=0.15')
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[src],
                           node_color=GOLD, node_size=800, node_shape='*')
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[mon],
                           node_color=PINK,  node_size=550, node_shape='D')
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=others,
                           node_color=ec,    node_size=280, alpha=0.75)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            font_color='white', font_size=7, font_weight='bold')
    ax.axis('off')


def plot_all(G, src, mon, res_m, res_g, k, stale_thresh):
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor(DARK)
    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.50, wspace=0.36,
                           left=0.06, right=0.97,
                           top=0.930, bottom=0.055)

    pos = nx.spring_layout(G, seed=7)

    # ── Compute static (no reselection) baselines ──────────────────────────
    init_m = matroidal_select(G, k, src)
    init_g = greedoidal_select(G, k, src)
    st_mt, st_ma = run_ctmc_static(G, src, mon, init_m)
    st_gt, st_ga = run_ctmc_static(G, src, mon, init_g)
    f_m_init = freshness(G, init_m, src)
    f_g_init = freshness(G, init_g, src)

    # ── Row 0: Final topology panels ──────────────────────────────────────
    draw_topo(fig.add_subplot(gs[0, 0]), G, pos, list(G.edges()),
              src, mon, GRAY,
              f"① Full Gossip Graph\n"
              f"{G.number_of_nodes()} nodes · {G.number_of_edges()} edges")

    draw_topo(fig.add_subplot(gs[0, 1]), G, pos, res_m['final_active'],
              src, mon, BLUE,
              f"② Matroidal  (final link set)\n"
              f"Reselections: {res_m['reselect_count']}  "
              f"f={res_m['freshness_log'][-1]:.4f}")

    draw_topo(fig.add_subplot(gs[0, 2]), G, pos, res_g['final_active'],
              src, mon, GREEN,
              f"③ Greedoidal  (final link set)\n"
              f"Reselections: {res_g['reselect_count']}  "
              f"f={res_g['freshness_log'][-1]:.4f}")

    # ── Row 1: CTMC age traces ─────────────────────────────────────────────
    ax_age = fig.add_subplot(gs[1, 0:3])
    sax(ax_age,
        "④ CTMC Version Age Trace  —  Adaptive vs Static  "
        "[Red dashed = sensor trigger threshold]",
        xlabel="Simulation Time (s)",
        ylabel="Version Age  (source − monitor)")

    for res, col, lbl in [
        (res_m, BLUE,   "Matroidal (adaptive)"),
        (res_g, GREEN,  "Greedoidal (adaptive)"),
    ]:
        ax_age.step(res['times'], res['ages'],
                    color=col, alpha=0.18, lw=0.5, where='post')
        xi, yi = rmean(res['ages'])
        ax_age.plot(res['times'][:len(yi)], yi,
                    color=col, lw=2.2, label=lbl)
        # Mark trigger events
        if res['trigger_times']:
            ax_age.scatter(res['trigger_times'], res['trigger_ages'],
                           color=col, s=60, zorder=5, marker='^',
                           edgecolors='white', linewidths=0.5, alpha=0.8)

    # Static baselines (thin)
    xi, yi = rmean(st_ma)
    ax_age.plot(st_mt[:len(yi)], yi,
                color=BLUE, lw=1.1, ls='--', alpha=0.45,
                label='Matroidal (static, no reselection)')
    xi, yi = rmean(st_ga)
    ax_age.plot(st_gt[:len(yi)], yi,
                color=GREEN, lw=1.1, ls='--', alpha=0.45,
                label='Greedoidal (static, no reselection)')

    ax_age.axhline(stale_thresh, color=RED, ls='--', lw=1.4, alpha=0.7,
                   label=f'Stale threshold = {stale_thresh}')
    ax_age.legend(facecolor=BG, edgecolor=GRID, labelcolor=GRAY, fontsize=8.5)

    # ── Row 2: Freshness evolution + trigger map ───────────────────────────
    ax_fr = fig.add_subplot(gs[2, 0:2])
    sax(ax_fr,
        "⑤ Freshness Score f(S) at Each Sensor Trigger  "
        "[SHS re-evaluated on every detection]",
        xlabel="Reselection Index",
        ylabel="Freshness f(S) = 1 / E[Age]")

    for res, col, lbl in [(res_m, BLUE, 'Matroidal'), (res_g, GREEN, 'Greedoidal')]:
        fl = res['freshness_log']
        ax_fr.plot(range(len(fl)), fl, 'o-', color=col, lw=2, ms=6, label=lbl)
        ax_fr.fill_between(range(len(fl)), fl, alpha=0.08, color=col)

    # Theoretical lower bounds
    f_all = freshness(G, list(G.edges()), src)
    ax_fr.axhline(f_all * (1 - 1/np.e), color=BLUE, ls=':', lw=1.3, alpha=0.55,
                  label=f"(1−1/e)·f_all = {f_all*(1-1/np.e):.4f}")
    ax_fr.axhline(f_all, color=ORANGE, ls='--', lw=1.2, alpha=0.55,
                  label=f"All-links f = {f_all:.4f}")
    ax_fr.legend(facecolor=BG, edgecolor=GRID, labelcolor=GRAY, fontsize=8.5)

    # ── Row 2 right: Trigger timing ────────────────────────────────────────
    ax_trig = fig.add_subplot(gs[2, 2])
    sax(ax_trig,
        "⑥ Sensor Trigger Timeline\n[When staleness threshold was hit]",
        xlabel="Simulation Time (s)",
        ylabel="Version Age at Trigger")

    for res, col, lbl in [(res_m, BLUE, 'Matroidal'), (res_g, GREEN, 'Greedoidal')]:
        if res['trigger_times']:
            ax_trig.scatter(res['trigger_times'], res['trigger_ages'],
                            color=col, s=70, label=lbl, zorder=4,
                            edgecolors='white', linewidths=0.5)
    ax_trig.axhline(stale_thresh, color=RED, ls='--', lw=1.2, alpha=0.7,
                    label=f'Threshold={stale_thresh}')
    ax_trig.legend(facecolor=BG, edgecolor=GRID, labelcolor=GRAY, fontsize=8.5)

    # ── Row 3: Per-step marginal gains + reselection count bar ────────────
    ax_mg = fig.add_subplot(gs[3, 0:2])
    sax(ax_mg,
        "⑦ Marginal Freshness Gain per Link Addition  "
        "[Final Selection — Diminishing Returns Verified]",
        xlabel="Step k",
        ylabel="Δ f(S)")

    def marginal_gains(G, edges, src):
        gains = []
        for i in range(len(edges)):
            prev = freshness(G, edges[:i], src)
            curr = freshness(G, edges[:i+1], src)
            gains.append(curr - prev)
        return gains

    w = 0.38
    mg_m = marginal_gains(G, res_m['final_active'], src)
    mg_g = marginal_gains(G, res_g['final_active'], src)
    xs = np.arange(1, max(len(mg_m), len(mg_g)) + 1)
    if mg_m:
        ax_mg.bar(xs[:len(mg_m)] - w/2, mg_m, width=w,
                  color=BLUE, alpha=0.80, label='Matroidal')
    if mg_g:
        ax_mg.bar(xs[:len(mg_g)] + w/2, mg_g, width=w,
                  color=GREEN, alpha=0.80, label='Greedoidal')
    ax_mg.legend(facecolor=BG, edgecolor=GRID, labelcolor=GRAY, fontsize=8.5)

    # ── Row 3 right: steady-state age comparison ───────────────────────────
    ax_bar = fig.add_subplot(gs[3, 2])
    sax(ax_bar,
        "⑧ Steady-State Version Age Comparison\n[Lower is Better]",
        ylabel="Mean Version Age (last 50%)")

    cut = lambda a: int(len(a) * 0.5)
    labels = ['Matroid\nAdaptive', 'Greedoid\nAdaptive',
              'Matroid\nStatic',   'Greedoid\nStatic']
    values = [
        np.mean(res_m['ages'][cut(res_m['ages']):]),
        np.mean(res_g['ages'][cut(res_g['ages']):]),
        np.mean(st_ma[cut(st_ma):]),
        np.mean(st_ga[cut(st_ga):]),
    ]
    cols   = [BLUE, GREEN, BLUE, GREEN]
    alphas = [0.85, 0.85, 0.45, 0.45]
    bars = []
    for lbl, val, col, alp in zip(labels, values, cols, alphas):
        b = ax_bar.bar(lbl, val, color=col, alpha=alp, edgecolor='white', linewidth=0.5)
        bars.append(b[0])
    for bar, val in zip(bars, values):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.05,
                    f'{val:.2f}', ha='center', va='bottom',
                    color='white', fontsize=8, fontweight='bold')
    ax_bar.tick_params(colors=GRAY, labelsize=7.5)

    # ── Legend patches ──────────────────────────────────────────────────────
    pats = [
        mpatches.Patch(color=GOLD,   label=f'Source (node {src})'),
        mpatches.Patch(color=PINK,   label=f'Monitor (node {mon})'),
        mpatches.Patch(color=BLUE,   label='Matroidal Greedy'),
        mpatches.Patch(color=GREEN,  label='Greedoidal Greedy'),
        mpatches.Patch(color=RED,    label=f'Stale threshold ({stale_thresh})'),
        mpatches.Patch(color=ORANGE, label='All-links oracle'),
    ]
    fig.legend(handles=pats, loc='upper right',
               bbox_to_anchor=(0.978, 0.968),
               facecolor=BG, edgecolor=GRID,
               labelcolor=GRAY, fontsize=8.5)

    mu_m = np.mean(res_m['ages'][cut(res_m['ages']):])
    mu_g = np.mean(res_g['ages'][cut(res_g['ages']):])
    winner = "Matroidal" if mu_m < mu_g else "Greedoidal"

    fig.suptitle(
        f"CTMC-Driven Adaptive Link Selection  ·  "
        f"Sensor-Triggered SHS Re-Scoring  ·  Matroidal vs Greedoidal Greedy\n"
        f"n={G.number_of_nodes()} nodes  m={G.number_of_edges()} edges  "
        f"k={k} active links  ·  "
        f"stale threshold={stale_thresh}  ·  "
        f"{winner} achieves lower steady-state age  "
        f"(Mat={mu_m:.2f}  Grd={mu_g:.2f})",
        color='white', fontsize=11, fontweight='bold', y=0.978
    )

    out = '/mnt/user-data/outputs/ctmc_adaptive_link_selection.png'
    plt.savefig(out, dpi=155, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    return fig


# ══════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 62)
    print("  CTMC-DRIVEN ADAPTIVE LINK SELECTION  (SHS + Sensor Loop)")
    print("=" * 62, "\n")

    G, src, mon = build_graph()
    K              = 6        # link budget per selection round
    STALE_THRESH   = 5        # age that triggers sensor alarm
    COOLDOWN       = 8.0      # min seconds between reselections
    T_MAX          = 300.0    # total simulation time

    print(f"Graph: {G.number_of_nodes()} nodes  {G.number_of_edges()} edges  "
          f"|  src={src}  mon={mon}  k={K}\n")

    print("Running CTMC (Matroidal adaptive) ...")
    res_m = run_ctmc_adaptive(G, src, mon, K,
                              algo='matroid',
                              stale_threshold=STALE_THRESH,
                              reselect_cooldown=COOLDOWN,
                              T_max=T_MAX, seed=42)

    print("Running CTMC (Greedoidal adaptive) ...")
    res_g = run_ctmc_adaptive(G, src, mon, K,
                              algo='greedoid',
                              stale_threshold=STALE_THRESH,
                              reselect_cooldown=COOLDOWN,
                              T_max=T_MAX, seed=42)

    print("\nPlotting ...")
    plot_all(G, src, mon, res_m, res_g, K, STALE_THRESH)
    print("Done.")