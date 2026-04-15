from ctmc import simulate_ctmc,Q
from graph_build import build_graph
from freshness import universal_freshness
from greedoidal_chosing import greedoidal
from matroidal_chosing import matroidal

G,src,mon=build_graph()
max_time=20
results=simulate_ctmc(Q,0,100)



