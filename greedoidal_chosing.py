import networkx as nx
from freshness import universal_freshness

def greedoidal(G,k,src,weight=0.4):
    """Greedoid: Only picks edges reachable from source."""
    sel, rem, fh, gh = [], list(G.edges()), [], []
    for _ in range(k):
        H = nx.DiGraph(); H.add_nodes_from(G.nodes()); H.add_edges_from(sel)
        reach = {src} | nx.descendants(H, src)
        frontier = [e for e in rem if e[0] in reach]
        if not frontier: break
        
        fc = universal_freshness(G, sel, src)
        best_score, best_e = -1.0, None
        for e in frontier:
            score = (universal_freshness(G, sel + [e], src) - fc)+(weight * G.edges[e]['rate'])
            if score > best_score:
                best_score, best_e = score, e
        if best_e is None: break
        sel.append(best_e); rem.remove(best_e)
        fh.append(universal_freshness(G, sel, src))
        gh.append(best_score)
    return sel, fh, gh