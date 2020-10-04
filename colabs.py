import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from funti import *


col = ['blue', 'green', 'c', 'm', 'y', 'k', "violet", "indigo"]
X = np.array([[28, 7], [36, 5], [32, 2], [56, 8], [47, 5], [50,100], [100,100], [26,59], [19,71],
              [75, 9], [34, 4], [56, 9], [28, 1], [33, 6]])
ncluster = plot_silh(X)
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

sink_node = get_sink_node_path(X, len(X))     # sink node creation


# print("Sink Node :\nX :",sink_node[0], "\nY :", sink_node[1])

def drawclusters():
    for i in range(ncluster):
        plt.scatter(X[y == i, 0], X[y == i, 1], s=100,
                    c=col[i], label=f'Cluster {i + 1}')
        art = mpatches.Circle(
            clusters_centroids[i], clusters_radii[i], edgecolor=col[i], fill=False)
        ax.add_patch(art)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200,
                c='red', label='Centroids', marker='x')
    plt.scatter(sink_node[0], sink_node[1], s=50,
                c='orange', label=f'Sink Node')


drawclusters()
plt.legend()
plt.tight_layout()
# plt.show()


no_of_nodes = len(X)
energies = {}
for i in X:
    energies[tuple(i)] = 5
# print(energies)
cluster_matrix = [[] for i in range(ncluster)]
for i in range(no_of_nodes):
  cluster_matrix[y[i]].append(X[i])


for i in range(len(sink_node[0])):
  present_sink_node = [sink_node[0][i],sink_node[1][i]]
  min_dist = 10000000000000
  cluster_no = -1
  for j in range(ncluster):
    dist  = get_distance(present_sink_node, centroids[j])
    if min_dist>=dist:
      min_dist = dist
      cluster_no = j
  optimal_point = get_optimal_node(present_sink_node,cluster_no, cluster_matrix, energies)
  drawclusters()
#   print(present_sink_node, optimal_point, cluster_no)
  ax.arrow(present_sink_node[0], present_sink_node[1], optimal_point[0] - present_sink_node[0], optimal_point[1] - present_sink_node[1],width=0.02,color='red',head_length=0.0,head_width=0.0)
  ax.scatter(present_sink_node[0], present_sink_node[1], s=50, c='red')
  ax.scatter(optimal_point[0], optimal_point[1], s=50, c='red')

plt.show()
