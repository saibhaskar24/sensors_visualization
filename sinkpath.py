import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from math import sqrt
import matplotlib.pyplot as plt

X = np.array([[28, 7], [36, 5], [32, 2], [56, 8], [47, 5],
              [75, 9], [34, 4], [56, 9], [28, 1], [33, 6]])
x1, x2, y = 1000000000, -1, -1
for i in X:
    if i[0] <= x1:
        x1 = i[0]
    if i[0] >= x2:
        x2 = i[0]
    if i[1] > y:
        y = i[1]
listOfPoints = [[x1-4, y], [x1, y+4], [x2, y+4], [x2+4, y]]
# print(listOfPoints)
listOfPoints.sort()
print(listOfPoints)
x_coOrdinates = []
y_coOrdinates = []
for i in listOfPoints:
    x_coOrdinates.append(i[0])
    y_coOrdinates.append(i[1])

x_coOrdinates = np.array(x_coOrdinates)
y_coOrdinates = np.array(y_coOrdinates)
print(x_coOrdinates)
print(y_coOrdinates)
sinkNode_x = np.linspace(x_coOrdinates.min(), x_coOrdinates.max(), 30)
spl = make_interp_spline(x_coOrdinates, y_coOrdinates, k=2)
sinkNode_y = spl(sinkNode_x)
print("x: ", sinkNode_x)
print("y: ", sinkNode_y)
#
