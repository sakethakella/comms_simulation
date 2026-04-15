import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from ctmc import simulate_ctmc,Q
from graph_build import build_graph
from freshness import universal_freshness
from greedoidal_chosing import greedoidal
from matroidal_chosing import matroidal

max_time=20
k=5#max number of links that can be activated

G,src,mon=build_graph()
results=simulate_ctmc(Q,0,100)
matroidal_edges,freshness_matroidal,best_score_matroidal=matroidal(G,k,src)
greedoidal_edges,freshness_greedoidal,best_score_greedoidal=greedoidal(G,k,src)
print(greedoidal_edges,matroidal_edges)
def build_graph1(edge_data):
    G = nx.DiGraph()
    G.add_nodes_from(range(10))
    src, mon = 0,9
    for u, v in edge_data:
        G.add_edge(u, v)
    return G

greedoidal_graph=build_graph1(greedoidal_edges)
Matroidal_graph=build_graph1(matroidal_edges)
#plots
plt.subplot(2,1,1)
nx.draw(greedoidal_graph,with_labels=True, node_color='lightblue', font_weight='bold')
plt.title('Greedoidal Selection')
plt.subplot(2,1,2)
nx.draw(Matroidal_graph,with_labels=True, node_color='lightgreen', font_weight='bold')
plt.title('Matroidal Selection')
plt.subplot(2,2,1)
# Add plot for freshness values
plt.plot(freshness_greedoidal, label='Greedoidal')
plt.plot(freshness_matroidal, label='Matroidal')
plt.xlabel('Time')
plt.ylabel('Freshness')
plt.title('Freshness Comparison')
plt.legend()
plt.subplot(2,2,2)
# Add plot for best scores
plt.plot(best_score_greedoidal, label='Greedoidal')
plt.plot(best_score_matroidal, label='Matroidal')
plt.xlabel('Time')
plt.ylabel('Best Score')
plt.title('Best Score Comparison')
plt.legend()
plt.tight_layout()
plt.show()






