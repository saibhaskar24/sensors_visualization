from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import silhouette_score


def get_sink_node_path():
    y = [y for y in range(0,40)]
    x = [(y/4)**2 for y in y]
    return [x,y]

def plot_silh(X):
    sil=[]
    for i in range(2,10):
        KM = KMeans(n_clusters=i,max_iter=500)
        KM.fit(X)
        lables = KM.labels_
        sil.append(silhouette_score(X,lables,metric='euclidean'))
    plt.clf()
    plt.plot(range(2,10),sil)
    plt.draw()
    print(sil.index(max(sil))+2)
    print(sil)
    return sil.index(max(sil))+2


col = ['blue', 'green', 'c', 'm', 'y', 'k', "violet", "indigo"]
X = np.array([[28, 7], [36, 5], [32, 2], [56, 8], [47, 5], [75, 9], [34, 4], [56, 9], [28, 1], [33, 6]])
ncluster=plot_silh(X)
kmeans = KMeans(n_clusters=ncluster,max_iter=500).fit(X)
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
    plt.scatter(X[y==i, 0], X[y==i, 1], s=100, c=col[i], label =f'Cluster {i + 1}')
    art = mpatches.Circle(clusters_centroids[i],clusters_radii[i], edgecolor=col[i],fill=False)
    ax.add_patch(art)
plt.scatter(centroids[:,0],centroids[:,1],s=200, c='red', label = 'Centroids',marker = 'x')
sink_node = get_sink_node_path()
print("Sink Node :\nX :",sink_node[0], "\nY :", sink_node[1])
plt.scatter(sink_node[0], sink_node[1], s=50, c='orange', label =f'Sink Node')
plt.legend()
plt.tight_layout()
plt.show()