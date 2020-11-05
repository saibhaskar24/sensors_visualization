import pandas as pd
dff = pd.ExcelFile('dara.xlsx')
k = []
for sheet in dff.sheet_names:
    df = pd.read_excel('data.xlsx',sheet)
    l = []
    for i in range(10):
        l.append(df['Node'+str(i+1)])
    k.append(l)
dff.close()