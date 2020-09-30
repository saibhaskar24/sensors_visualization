import numpy as np
from math import sqrt, pow
sink_node = [93, 23]
X = np.array([[28, 7], [36, 5], [32, 2], [56, 8], [47, 5],
              [75, 9], [34, 4], [56, 9], [28, 1], [33, 6]])
no_of_nodes = len(X)
energies = {}
for i in X:
    energies[tuple(i)] = 5


def get_distance(x, y):
    return sqrt(pow((x[0]-y[0]), 2) + pow((x[1]-y[1]), 2))


def get_energy_of_tramission(sink_node, cluster_node):
    data_agg_energy = 5 * 10 ** -9
    tx_fs_energy = 10 * 10 ** -12
    tx_energy = 50 * 10 ** -9
    distance = get_distance(sink_node, cluster_node)
    return (5000*(tx_energy+data_agg_energy)-(5000*tx_fs_energy*(distance**2)))


def txEnergyChange(sink_node, cluster_node):
    tx_energy = get_energy_of_tramission(sink_node, cluster_node)
    if tx_energy <= energies[tuple(cluster_node)]:
        energies[tuple(cluster_node)] -= tx_energy
        print(
            f"From cluster node: {cluster_node} Energy of that node remaining: {energies[tuple(cluster_node)]} ")
        return 1
    else:
        print("Dark Hole occured")
        return 0


for i in X:
    temp = txEnergyChange(sink_node, i)
