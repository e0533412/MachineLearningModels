import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

df_wine = pd.read_csv('../data/winequalityN.csv')
df_wine.fillna(0.0,inplace = True)

#convert type to 0,1
def convertToNum(x):
    if x == "white":
        return 0
    else:
        return 1
df_wine["type"] = df_wine["type"].apply(convertToNum)

x = df_wine.iloc[:, 1:13].values
dbscan = DBSCAN(eps=5, min_samples = 50)
clusters = dbscan.fit_predict(x)

# colors = [
#     'red', 'blue', 'green', 'chocolate',
#     'lightgreen', 'cornflowerblue',
#     'violet', 'darkturquoise', 'lightskyblue',
#     'magenta', 'yellowgreen', 'coral',
#     'darkorange', 'mediumpurple',
#     'olive', 'tan',
# ]

print(np.unique(clusters))
#2D
for i in np.unique(clusters):
    label = 'Outlier' if i == -1 else 'Cluster ' + str(i + 1)
    plt.scatter(x[clusters==i,0], x[clusters==i,10],
               label=label)

plt.xlabel(df_wine.columns[1])
plt.ylabel(df_wine.columns[11])
plt.legend()
plt.title('Db Scan_wine 2D')
plt.show()

#3D
fig = plt.figure(figsize= (10,7))
ax = plt.axes(projection = '3d')

print(np.unique(clusters))

for i in np.unique(clusters):
    label = 'Outlier' if i == -1 else 'Cluster ' + str(i + 1)
    ax.scatter3D(x[clusters==i,0], x[clusters==i,7],
                x[clusters == i,10 ],
                 label=label)

ax.set_xlabel(df_wine.columns[1])
ax.set_ylabel(df_wine.columns[8])
ax.set_zlabel(df_wine.columns[11])
plt.legend()
plt.title("DBScan Clustering")
plt.show()