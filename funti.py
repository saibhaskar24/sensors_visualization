from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from math import sqrt, pow
import numpy as np


def getpathpoints(parent,n,g,centroids):
    path = [[],[]]
    # print( "Edge \tWeight")
    for i in range(1, n):
        frames = np.linspace(0, 1, num=int(g[i][parent[i]]))
        for t in frames:
            x = centroids[parent[i]][0] + (centroids[i][0]-centroids[parent[i]][0]) * t
            y = centroids[parent[i]][1] + (centroids[i][1]-centroids[parent[i]][1]) * t
            path[0].append(x)
            path[1].append(y)
        # print (parent[i], "-", i, "\t", g[i][parent[i]])
    return path
    
def minKey(key, mstSet,n):
        min = float('inf')
        for v in range(n):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v
        return min_index

def primMST(n,g):
        key = [float('inf')] * n
        parent = [None] * n 
        key[0] = 0
        mstSet = [False] * n
        parent[0] = -1
        for cout in range(n):
            u = minKey(key, mstSet,n)
            mstSet[u] = True
            for v in range(n):
                if g[u][v] > 0 and mstSet[v] == False and key[v] > g[u][v]:
                        key[v] = g[u][v]
                        parent[v] = u
        return parent

# def get_sink_node_path():
#     y = [y for y in range(0,40)]
#     x = [(y/4)**2 for y in y]
#     return [x,y]


def get_sink_node_path(centroids, n):
    graph = creategraph(centroids, n)
    # print(graph)
    parent = primMST(n,graph)
    # print(parent)
    return getpathpoints(parent,n,graph,centroids)





def plot_silh(X):
    sil=[]
    for i in range(2,10):
        KM = KMeans(n_clusters=i,max_iter=500)
        KM.fit(X)
        lables = KM.labels_
        sil.append(silhouette_score(X,lables,metric='euclidean'))
    # plt.clf()
    # plt.plot(range(2,10),sil)
    # plt.draw()
    print(sil.index(max(sil))+2)
    print(sil)
    return sil.index(max(sil))+2


def get_distance(x, y):
    return sqrt( pow((x[0]-y[0]), 2) + pow((x[1]-y[1]), 2))


def get_energy_of_tramission(sink_node, cluster_node):
    data_agg_energy = 5 * 10 ** -9
    tx_fs_energy = 10 * 10 ** -12
    tx_energy = 50 * 10 ** -9
    distance = get_distance(sink_node, cluster_node)
    return (5000*(tx_energy+data_agg_energy)-(5000*tx_fs_energy*(distance**2)))



def get_optimal_node(sink_node, min_dist_cluster_no, cluster_matrix, energies):
    tx_energy = {}
    max_energy = -1
    for i in cluster_matrix[min_dist_cluster_no]:
        tx_energy[tuple(i)] = get_energy_of_tramission(sink_node, i)
        max_energy = max(max_energy, energies[tuple(i)])
    optimalNode = []
    min_tx_energy = 10000000000000000000
    for i in tx_energy:
        if energies[i] == max_energy:
            if min_tx_energy >= tx_energy[i]:
                optimalNode = list(i)
                min_tx_energy = tx_energy[i]
    energies[tuple(optimalNode)] -= min_tx_energy
    return optimalNode



def creategraph(centroids, n):
    g = []
    for i in range(n):
        k = []
        for j in range(n):
            if i == j:
                k.append(0)
            else:
                k.append(get_distance(centroids[i], centroids[j]))
        g.append(k)
    return g