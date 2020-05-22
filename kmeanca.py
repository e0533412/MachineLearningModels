import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

#Function to map type
def myfunction(x):
    if x == "red":
        return 0
    else:  # white
        return 1

#readcsv
df = pd.read_csv('data/winequalityN.csv')

#applyfunctiontomaptype
df["type"] = df["type"].apply(myfunction)

#fillnullvalueswithzero
df = df.fillna(0)

x = df.iloc[:, :12].values

#Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 7)
clusters = kmeans.fit_predict(x)

#3Dplot
colors = 'rgbkcmy'
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection='3d')

for i in np.unique(clusters):
    ax.scatter3D(x[clusters==i,1],
             x[clusters==i,3],
             x[clusters==i,11],
                 color=colors[i], label='Cluster ' + str(i + 1))

ax.set_xlabel(df.columns[1])
ax.set_ylabel(df.columns[3])
ax.set_zlabel(df.columns[11])

plt.legend()
plt.title('K-Means Clustering')
plt.show()
