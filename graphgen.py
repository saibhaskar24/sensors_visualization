import networkx as nx
from math import sqrt, pow
import matplotlib.pyplot as plt
# from networkx.algorithms.shortest_paths.generic import shortest_path, shortest_path_length
from networkx.algorithms.shortest_paths.weighted import dijkstra_path_length,dijkstra_path
def get_distance(x, y):
    return sqrt(pow((x[0]-y[0]), 2) + pow((x[1]-y[1]), 2))

def gengraph(points,limit):
    G = nx.Graph()
    for point in points:
        G.add_node(tuple(point),pos=tuple(point))
    for point in points:
        for secpoint in points:
            distance = get_distance(point, secpoint)
            if distance > limit or distance == 0:
                continue
            G.add_edge(tuple(point), tuple(secpoint), weight=distance)
    # print(G.edges(data=True))
    # print(len(G.nodes),len(points))
    return G

G = gengraph([[28, 7], [36, 5], [32, 2], [56, 8], [47, 5], [50, 100], [100, 100], [26, 59], [19, 71],
              [75, 9], [34, 4], [56, 9], [28, 1], [33, 6]] , 100)
pos = nx.get_node_attributes(G,'pos')
nx.draw(G,pos)
# path = shortest_path(G,(28,7),(100,100))
# lenght = shortest_path_length(G,(28,7),(100,100))
algo = dijkstra_path_length(G,(28,7),(100,100))
dalgo = dijkstra_path(G,(28,7),(100,100))
print(algo,dalgo)
# # G = nx.petersen_graph()
# plt.subplot(121)
# nx.draw(G, with_labels=True, font_weight='bold')
# plt.subplot(122)
# nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
plt.show()

#https://networkx.org/documentation/networkx-2.1/reference/algorithms/shortest_paths.html