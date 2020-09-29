from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def get_sink_node_path():
    y = [y for y in range(0,40)]
    x = [(y/4)**2 for y in y]
    return [x,y]


ncluster = int(input("Enter number of clusters : "))
col = ['blue', 'green', 'c', 'm', 'y', 'k']
X = np.array([[28, 7], [36, 5], [32, 2], [56, 8], [47, 5], [75, 9], [34, 4], [56, 9], [28, 1], [33, 6]])
kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(X)
y = kmeans.labels_
centroids = kmeans.cluster_centers_
print("Clusters :",y)
clusters_centroids=dict()
clusters_radii= dict()
for cluster in range(ncluster):
    clusters_centroids[cluster]=list(zip(centroids[:, 0],centroids[:,1]))[cluster]
    clusters_radii[cluster] = max([np.linalg.norm(np.subtract(i,clusters_centroids[cluster])) for i in zip(X[y == cluster, 0],X[y == cluster, 1])])
print("Centroids :", clusters_centroids)
print("Redii's :", clusters_radii)
fig, ax = plt.subplots(1,figsize=(7,5))
for i in range(ncluster):
    plt.scatter(X[y==i, 0], X[y==i, 1], s=100, c=col[i], label =f'Cluster {i}')
    art = mpatches.Circle(clusters_centroids[i],clusters_radii[i], edgecolor=col[i],fill=False)
    ax.add_patch(art)
plt.scatter(centroids[:,0],centroids[:,1],s=200, c='red', label = 'Centroids',marker = 'x')
sink_node = get_sink_node_path()
print("Sink Node :\nX :",sink_node[0], "\nY :", sink_node[1])
plt.scatter(sink_node[0], sink_node[1], s=50, c='orange', label =f'Sink Node')
plt.legend()
plt.tight_layout()
plt.show()