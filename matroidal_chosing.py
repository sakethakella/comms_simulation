import numpy as np
import networkx as nx
from freshness import universal_freshness

def matroidal(G,k,src,weight=0.2):
    sel, rem, fh, gh = [], list(G.edges()), [], []
    for _ in range(k):
        fc = universal_freshness(G, sel, src)
        best_score, best_e = -1.0, None
        for e in rem:
            score = (universal_freshness(G, sel + [e], src) - fc)+(weight * G.edges[e]['rate'])
            if score > best_score:
                best_score, best_e = score, e
        if best_e is None: break
        sel.append(best_e); rem.remove(best_e)
        fh.append(universal_freshness(G, sel, src))
        gh.append(best_score)
    return sel, fh, gh