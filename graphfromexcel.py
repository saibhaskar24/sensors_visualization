import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dff = pd.ExcelFile('data.xlsx')
d = {}
for i in range(10):
    d['Node'+str(i+1)] = []
for sheet in dff.sheet_names:
    df = pd.read_excel('data.xlsx',sheet)
    for i in range(10):
        d['Node'+str(i+1)].append(df['Node'+str(i+1)])
# print(d['Node1'][0].tolist())
# print(d['Node1'][0][1])
dff.close()

col = ['b--', 'r--', 'm', 'y', 'k', "violet", "indigo"]
name = ["Static Cluster Head", "Dynamic Cluster Path (Proposed)"]
i = 0
figure, axes = plt.subplots(nrows=2, ncols=5)
for rows in axes:
    for ax1 in rows:
        for j in range(len(d['Node'+str(i+1)])):
            L = d['Node'+str(i+1)][j].tolist()
            l = []
            for I in range(0,len(L),200):
                l.append(L[I])
            ax1.plot( range(len(l)),l, col[j],label = name[j%3])
            ax1.set_title('Node'+str(i+1))
            ax1.legend()
            
        i+=1
plt.show()