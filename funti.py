from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from math import sqrt, pow
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline, pchip_interpolate
from scipy.spatial import ConvexHull
import networkx as nx


def getpathpoints(parent, n, g, centroids):
    path = [[], []]
    # print( "Edge \tWeight")
    for i in range(1, n):
        frames = np.linspace(0, 1, num=int(g[i][parent[i]]))
        for t in frames:
            x = centroids[parent[i]][0] + \
                (centroids[i][0]-centroids[parent[i]][0]) * t
            y = centroids[parent[i]][1] + \
                (centroids[i][1]-centroids[parent[i]][1]) * t
            path[0].append(x)
            path[1].append(y)
        # print (parent[i], "-", i, "\t", g[i][parent[i]])
    return path


def minKey(key, mstSet, n):
    min = float('inf')
    for v in range(n):
        if key[v] < min and mstSet[v] == False:
            min = key[v]
            min_index = v
    return min_index


def primMST(n, g):
    key = [float('inf')] * n
    parent = [None] * n
    key[0] = 0
    mstSet = [False] * n
    parent[0] = -1
    for _ in range(n):
        u = minKey(key, mstSet, n)
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


# def get_sink_node_path(centroids, n):
#     graph = creategraph(centroids, n)
#     # print(graph)
#     parent = primMST(n,graph)
#     # print(parent)
#     return getpathpoints(parent,n,graph,centroids)


def geteratepointsinbetween(p1, p2):
    frames = np.linspace(0, 1, num=int(get_distance(p1, p2))//3)
    path = [[], []]
    for t in frames:
        x = p1[0] + \
            (p2[0] - p1[0]) * t
        y = p1[1] + \
            (p2[1] - p1[1]) * t
        path[0].append(x)
        path[1].append(y)
    return path


def get_sink_node_path(X, n, temp_dist):
    # temp_dist = 100
    x1, x2, y2, y1 = [10000000000, -1], [-1, -
                                         1], [-1, -1], [-1, 10000000000000]
    for i in X:
        if i[0] <= x1[0]:
            x1 = list(i)
        if i[0] >= x2[0]:
            x2 = list(i)
        if i[1] >= y2[1]:
            y2 = list(i)
        if i[1] <= y1[1]:
            y1 = list(i)

    x1[0] -= 3
    x2[0] += 3
    y2[1] += 3
    # p1, p2 = [x1[0], (x1[1]+y1[1])//2], [x2[0], (x2[1]+y1[1])//2]

    if x1[0] == y2[0]:
        y2[0] += 1
    if y2[0] == x2[0]:
        y2[0] += 1

    hull = ConvexHull(X)

    def isnotInHull(point, tolerance=1e-12):
        return all(
            (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
            for eq in hull.equations)

    vx, vy = [], []
    for i in hull.vertices:
        vx.append(X[i][0])
        vy.append(X[i][1]+7)

    pointsx = []
    pointsy = []
    for i in range(len(vy)-1):
        gen = geteratepointsinbetween([vx[i], vy[i]], [vx[i+1], vy[i+1]])
        pointsx += gen[0]
        pointsy += gen[1]
    gen = geteratepointsinbetween([vx[0], vy[0]], [vx[-1], vy[-1]])
    pointsx += gen[0]
    pointsy += gen[1]
    i = 0
    while(i < len(pointsx)):
        if isnotInHull((pointsx[i], pointsy[i])):
            pointsx.pop(i)
            pointsy.pop(i)
            i -= 1
        i += 1

    sinkNode_x = pointsx[::]
    sinkNode_y = pointsy[::]
    BaseStation = [(min(sinkNode_x)+max(sinkNode_x))//2, max(sinkNode_y)+4]
    i = 0
    leng = len(sinkNode_x)
    while i < leng:
        if get_distance(BaseStation, [sinkNode_x[i], sinkNode_y[i]]) > temp_dist:
            sinkNode_x.pop(i)
            sinkNode_y.pop(i)
            leng -= 1
        else:
            i += 1
    return [sinkNode_x, sinkNode_y], BaseStation


def plot_silh(X):
    sil = []
    for i in range(2, 10):
        KM = KMeans(n_clusters=i, max_iter=500)
        KM.fit(X)
        lables = KM.labels_
        sil.append(silhouette_score(X, lables, metric='euclidean'))
    print(sil.index(max(sil))+2)
    print(sil)
    return sil.index(max(sil))+2


def get_distance(x, y):
    return sqrt(pow((x[0]-y[0]), 2) + pow((x[1]-y[1]), 2))


def get_energy_of_tramission(sink_node, cluster_node):
    data_agg_energy = 5 * 10 ** -9
    tx_fs_energy = 10 * 10 ** -12
    tx_energy = 50 * 10 ** -9
    distance = get_distance(sink_node, cluster_node)
    return (5000*(tx_energy+data_agg_energy)-(5000*tx_fs_energy*(distance**2)))


def get_optimal_node(sink_node, min_dist_cluster_no, cluster_matrix, energies, temp_dist):
    # temp_dist = 20
    tx_energy = {}
    max_energy = -1

    list_of_min_nodes = cluster_matrix[min_dist_cluster_no][::]

    i = 0
    leng = len(list_of_min_nodes)
    while i < leng:
        if get_distance(list_of_min_nodes[i], sink_node) > temp_dist:
            list_of_min_nodes.pop(i)
            leng -= 1
        else:
            i += 1

    if len(list_of_min_nodes) == 0:
        return sink_node
    for i in list_of_min_nodes:
        tx_energy[tuple(i)] = get_energy_of_tramission(sink_node, i)
        max_energy = max(max_energy, energies[tuple(i)])
    c = 0
    for i in list_of_min_nodes:
        if energies[tuple(i)] >= tx_energy[tuple(i)]:
            break
        c += 1
    if c == len(list_of_min_nodes):
        return -1
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


def staticSinkPath():
    y = [y for y in range(0, 31)]
    x = [(y/4)**2 for y in y]
    return [x, y]


def optimalToOptimal(optimalNode, group, temp_dist, energies):
    tx_energy = {}
    max_energy = -1
    i = 0

    leng = len(group)
    while i < leng:
        if get_distance(group[i], optimalNode) > temp_dist:
            group.pop(i)
            leng -= 1
        else:
            i += 1
    if len(group) == 0:
        return optimalNode
    for i in group:
        tx_energy[tuple(i)] = get_energy_of_tramission(sink_node, i)
        max_energy = max(max_energy, energies[tuple(i)])
    c = 0
    for i in group:
        if energies[tuple(i)] >= tx_energy[tuple(i)]:
            break
        c += 1
    if c == len(group):
        return -1
    optimalNode = []
    min_tx_energy = 10000000000000000000
    for i in tx_energy:
        if energies[i] == max_energy:
            if min_tx_energy >= tx_energy[i]:
                optimalNode = list(i)
                min_tx_energy = tx_energy[i]
    energies[tuple(optimalNode)] -= min_tx_energy
    return optimalNode

from networkx.algorithms.shortest_paths.weighted import dijkstra_path_length,dijkstra_path

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
    return G

def interpoint(clusterpoints,limit,optimalpointincluster):
    G = gengraph(clusterpoints,limit)
    d = {}
    for i in clusterpoints:
        if optimalpointincluster != i:
            path = dijkstra_path_length(G,optimalpointincluster,i)
            cost = dijkstra_path(G,optimalpointincluster,i)
            d[i] = (path,cost)
    return d

def expernalpoint(individualclusteroptimalpoints,limit,optimalpoint):
    G = gengraph(individualclusteroptimalpoints,limit)
    d = {}
    for i in individualclusteroptimalpoints:
        if optimalpoint != i:
            path = dijkstra_path_length(G,optimalpoint,i)
            cost = dijkstra_path(G,optimalpoint,i)
            d[i] = (path,cost)
    return d