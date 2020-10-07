import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from celluloid import Camera
from funti import *


col = ['blue', 'green', 'c', 'm', 'y', 'k', "violet", "indigo"]
X = np.array([[28, 7], [36, 5], [32, 2], [56, 8], [47, 5], [50, 100], [100, 100], [26, 59], [19, 71],
              [75, 9], [34, 4], [56, 9], [28, 1], [33, 6]])

X = np.array([[49, 10], [51, 16], [53, 13], [55, 12], [91, 15], [84, 17], [92, 32], [97, 11], [63, 85], [66, 85], [68, 92], [
             70, 86], [62, 76], [55, 78], [103, 119], [104, 110], [100, 108], [104, 105], [78, 56], [76, 58], [79, 57], [71, 46]])
ncluster = plot_silh(X)
maxrange_basestation = int(input("Entr max range of Base Station : "))
maxrange_node = int(input("Entr max range of each sensor : "))
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

sink_node, BaseStation = get_sink_node_path(X, len(X), maxrange_basestation)     # sink node creation


# print("Sink Node :\nX :",sink_node[0], "\nY :", sink_node[1])

def drawclusters():
    for i in range(ncluster):
        points = X[y == i]
        ax.scatter(points[:, 0], points[:, 1], s=100,
                   c=col[i], label=f'Cluster {i + 1}')
        hull = ConvexHull(points)
        # close the polygon by appending the first point at the end
        vert = np.append(hull.vertices, hull.vertices[0])
        ax.plot(points[vert, 0], points[vert, 1], '--', c=col[i])
        ax.fill(points[vert, 0], points[vert, 1], c=col[i], alpha=0.2)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200,
                c='red', label='Centroids', marker='x')
    plt.scatter(sink_node[0], sink_node[1], s=60,
                c='orange', label=f'Sink Node')
    ax.scatter(BaseStation[0], BaseStation[1], s=500, c='black')
    circlebase = plt.Circle(BaseStation, maxrange_basestation, color='black', clip_on=False, alpha=0.1)
    ax.add_artist(circlebase)



drawclusters()
plt.legend()
# plt.show()

camera = Camera(fig)


no_of_nodes = len(X)
energies = {}
for i in X:
    energies[tuple(i)] = 5
# print(energies)
cluster_matrix = [[] for i in range(ncluster)]
for i in range(no_of_nodes):
  cluster_matrix[y[i]].append(list(X[i]))

tempClusterNode = [36, 5]

for i in range(len(sink_node[0])):
    present_sink_node = [sink_node[0][i], sink_node[1][i]]
    min_dist = 10000000000000
    cluster_no = -1
    for j in range(ncluster):
        dist = get_distance(present_sink_node, centroids[j])
        if min_dist >= dist:
            min_dist = dist
            cluster_no = j
    optimal_point = get_optimal_node(
        present_sink_node, cluster_no, cluster_matrix, energies, maxrange_node)
    optimalCluster_Cluster = []
    if optimal_point != present_sink_node:
        if tempClusterNode not in cluster_matrix[cluster_no]:
            clusterPoints = cluster_matrix[cluster_no]
            minDist = 10000000
            # print(clusterPoints)
            for i in clusterPoints:
                dist = get_distance(i, tempClusterNode)
                if minDist > dist:
                    minDist = dist
                    optimalCluster_Cluster = i
            # print("Signal ",optimalCluster_Cluster)
#   print(present_sink_node, optimal_point, cluster_no)
    ax.arrow(present_sink_node[0], present_sink_node[1], optimal_point[0] - present_sink_node[0], optimal_point[1] - present_sink_node[1],width=0.02,color='red',head_length=0.0,head_width=0.0)
    ax.scatter(present_sink_node[0], present_sink_node[1], s=10, c='red')
    ax.scatter(optimal_point[0], optimal_point[1], s=10, c='red')
  # plt.show()

  # camera.snap()
  # drawclusters()

plt.show()
# camera.snap()
# animation = camera.animate()

# print(animation)
# print("Writing ...")
# animation.save("m.mp4",fps=1)
# print("Done")
# plt.show()
