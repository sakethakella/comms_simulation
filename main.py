from ctmc import simulate_ctmc,Q
from graph_build import build_graph

G,src,mon=build_graph()
max_time=20
results=simulate_ctmc(Q,0,100)

print(G[0])
