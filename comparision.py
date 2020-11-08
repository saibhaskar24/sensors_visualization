import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from celluloid import Camera
from funti import *

import csv
f = open("sinkpath3.csv", "w", newline='')
writer = csv.writer(f)
writer.writerow(["Node1", "Node2", "Node3", "Node4", "Node5",
                 "Node6", "Node7", "Node8", "Node9", "Node10"])

col = ['blue', 'green', 'c', 'm', 'y', 'k', "violet", "indigo"]
X = np.array([[28, 7], [36, 5], [32, 2], [56, 8], [47, 5],
              [75, 9], [34, 4], [56, 9], [28, 1], [33, 6]])
ncluster = plot_silh(X)
maxrange_basestation = int(input("Enter max range of Base Station : "))
maxrange_node = int(input("Enter max range of each sensor : "))
kmeans = KMeans(n_clusters=ncluster, max_iter=500).fit(X)
y = kmeans.labels_
centroids = kmeans.cluster_centers_
print("Clusters :", y)
clusters_centroids = dict()
clusters_radii = dict()
for cluster in range(ncluster):
    clusters_centroids[cluster] = list(
        zip(centroids[:, 0], centroids[:, 1]))[cluster]
    clusters_radii[cluster] = max([np.linalg.norm(np.subtract(
        i, clusters_centroids[cluster])) for i in zip(X[y == cluster, 0], X[y == cluster, 1])])
print("Centroids :", clusters_centroids)
print("Redii's :", clusters_radii)

fig, ax = plt.subplots(1, figsize=(7, 5))

sink_node1, BaseStation1 = get_sink_node_path(
    X, len(X), maxrange_basestation)     # sink node creation
# sink_node2 = staticSinkPath()
# p1 = []
# p2 = []
# for i in X:
#     p1.append(i[0])
#     p2.append(i[1])
# print(p1, p2)
# base = [-1, -1]

# base[0] = (min(p1)+max(p1))//2
# base[1] = max(p2)+5

# # Energy Utilisation
# clusterPoints = [[] for i in range(ncluster)]
# for i in range(len(X)):
#     clusterPoints[y[i]].append(X[i])
# final = []
# for i in range(ncluster):
#     min1 = 10000000000
#     ind = [-1, -1]
#     p = clusterPoints[i]
#     for i in p:
#         s = 0
#         for j in p:
#             d = get_distance(i, j)
#             s += d
#         s /= len(X)
#         if min1 > s:
#             min1 = s
#             ind = i
# final.append(list(i))
# print(final)
# best = [-1, -1]
# mini = 10000000000
# for i in final:
#     d = get_distance(i, base)
#     if mini > d:
#         mini = d
#         best = i
no_of_nodes = len(X)
energies1 = {}
energies2 = {}
for i in X:
    energies1[tuple(i)] = 2
    energies2[tuple(i)] = 2


def get_energy_of_tramission(sink_node, cluster_node):
    data_agg_energy = 5 * 10 ** -9
    tx_fs_energy = 10 * 10 ** -12
    tx_energy = 50 * 10 ** -9
    distance = get_distance(sink_node, cluster_node)
    return (5000*(tx_energy+data_agg_energy)-(5000*tx_fs_energy*(distance**2)))


def txEnergyChange(sink_node, optNode, energies):
    tx_energy = get_energy_of_tramission(sink_node, optNode)
    if tx_energy <= energies[tuple(optNode)]:
        energies[tuple(optNode)] -= tx_energy
        return 1
    else:
        print("Dark Hole occured")
        return 0


# count = 0

# while(2):
#     count += 1
#     temp = 1
#     for i in range(len(sink_node2[0])):
#         present_sink_node = [sink_node2[0][i], sink_node2[1][i]]
#         min_dist = 10000000000000
#         optNode = -1
#         for j in range(no_of_nodes):
#             dist = get_distance(present_sink_node, X[j])
#             if min_dist >= dist:
#                 min_dist = dist
#                 optNode = j
#         temp = txEnergyChange(present_sink_node, X[optNode], energies1)
#         if temp == 0:
#             break
#     writer.writerow(list(energies1.values()))
#     if temp == 0:
#         break
# print(count)

# count = 0

# while(2):
#     count += 1
#     temp = 1
#     present_sink_node = base
#     min_dist = 10000000000000
#     optNode = -1
#     for j in range(no_of_nodes):
#         dist = get_distance(present_sink_node, X[j])
#         if min_dist >= dist:
#             min_dist = dist
#             optNode = j
#     temp = txEnergyChange(present_sink_node, X[optNode], energies1)
#     if temp == 0:
#         break
#     writer.writerow(list(energies1.values()))
#     if temp == 0:
#         break
# print(count)
count = 0
cluster_matrix = [[] for i in range(ncluster)]

for i in range(no_of_nodes):
    cluster_matrix[y[i]].append(list(X[i]))
while(2):
    count += 1
    for i in range(len(sink_node1[0])):
        present_sink_node = [sink_node1[0][i], sink_node1[1][i]]
        min_dist = 10000000000000
        cluster_no = -1
        for j in range(ncluster):
            dist = get_distance(present_sink_node, centroids[j])
            if min_dist >= dist:
                min_dist = dist
                cluster_no = j
        optimal_point = get_optimal_node(
            present_sink_node, cluster_no, cluster_matrix, energies2, maxrange_node)
        if optimal_point == -1:
            break
    writer.writerow(energies2.values())
    if count>=100000:
        break
print(count)

# count = 0
# while(1):
#     count += 1
#     d = get_energy_of_tramission(base, best)
#     if d <= energies1[tuple(best)]:
#         energies1[tuple(best)] -= d
#     else:
#         break
#     writer.writerow(list(energies1.values()))


f.close()
