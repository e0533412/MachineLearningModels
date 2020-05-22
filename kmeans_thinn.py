import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


df_wine = pd.read_csv('../data/winequalityN.csv')
df_wine.fillna(0.0,inplace = True)
x = df_wine.iloc[:,1:13].values

kmeans = KMeans(n_clusters =2)
clusters = kmeans.fit_predict(x)

colors = 'rgbkcmy'
#3D plot
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection='3d')

for i in np.unique(clusters):
    ax.scatter3D(x[clusters==i,0],
             x[clusters==i,1],
             x[clusters==i,2],
                 color=colors[i], label='Cluster ' + str(i + 1))

ax.set_xlabel(df_wine.columns[1])
ax.set_ylabel(df_wine.columns[2])
ax.set_zlabel(df_wine.columns[3])

plt.legend()
plt.title('K-Means Clustering')
plt.show()

# optimum number of clusters for k-means
wcss = []

# Trying kmeans for k=1 to k=10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# line graph,to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Actual Categorization
species = np.reshape(df_wine.loc[:, ['type']].values, (-1,))
i = 0
for label in np.unique(species):
    plt.scatter(x[species==label,0], x[species==label,1],
                color=colors[i], label=label)
    i += 1

plt.legend()
plt.title('Wine Type')
plt.xlabel(df_wine.columns[10])
plt.ylabel(df_wine.columns[11])
plt.show()