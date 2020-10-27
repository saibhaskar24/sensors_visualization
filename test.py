# from scipy.spatial import ConvexHull, convex_hull_plot_2d
# import numpy as np
# from scipy.interpolate import pchip_interpolate
# points = np.array([[49, 10], [51, 16], [53, 13], [55, 12], [91, 15], [84, 17], [92, 32], [97, 11], [63, 85], [66, 85], [68, 92], [
#              70, 86], [62, 76], [55, 78], [103, 119], [104, 110], [100, 108], [104, 105], [78, 56], [76, 58], [79, 57], [71, 46]])
# hull = ConvexHull(points = points)

# def isInHull(point, tolerance=1e-12):
#     return all(
#         (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
#         for eq in hull.equations)

# vx,vy = [],[]
# x , y = [], []
# for i in hull.vertices:
#     x.append(points[i][0]-3)
#     vx.append(points[i][0])
#     x.append(points[i][0]+3)
#     y.append(points[i][1]+3)
#     vy.append(points[i][1])
#     y.append(points[i][1]+3)

# d = {}
# for i in range(len(x)):
#     if x[i] in d:
#         if d[x[i]] < y[i]:
#             d[x[i]] = y[i]
#     else:
#         d[x[i]] = y[i]

# x,y = [],[]
# for i in sorted(d.keys()):
#     x.append(i)
#     y.append(d[i])

# i = 0
# print(x)
# print(y)
# while(i<len(x)):
#     if isInHull((x[i],y[i])):
#         x.pop(i)
#         y.pop(i)
#         i-=1
#     i+=1
# print(x)
# print(y)
# print(vx,vy)
# x2 = np.linspace(min(x), max(x), 100)
# y2 = pchip_interpolate(x, y, x2)
# print(x2,y2)
# i = 0
# x2=list(x2)
# y2=list(y2)
# while(i<len(x2)):
#     if isInHull((x2[i],y2[i])):
#         x2.pop(i)
#         y2.pop(i)
#         i-=1
#     i+=1
# print(x2,y2)


# import matplotlib.pyplot as plt
# plt.plot(points[:,0], points[:,1], 'o')
# for simplex in hull.simplices:
#     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
# plt.plot(x2, y2)
# plt.plot(x, y, "o")
# plt.show()



def printPowerSet(arr, n,r): 
    ans = 0
    _list = [] 
    for i in range(2**n): 
        subset = "" 
        for j in range(n): 
            if (i & (1 << j)) != 0: 
                subset += str(arr[j]) + "|"
        if subset not in _list and len(subset) > 0: 
            _list.append(subset) 
    for subset in _list: 
        arr = subset.split('|') 
        l = []
        for string in arr: 
            # print(string)
            if string != '': 
               l.append(int(string))
        if sum(l) < r:
            ans+=1
    print(ans) 

n = int(input())
r = int(input())
a = list(map(int,input().split()))

printPowerSet(a, n,r) 

