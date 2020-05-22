import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

#Part 1
dataset = pd.read_csv('winequality-white.csv')
x = dataset.iloc[:, [0,1,2]].values

#Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3)
clusters = kmeans.fit_predict(x)

colors = 'rgbkcmy'

ax=plt.subplot(111,projection='3d')

for i in np.unique(clusters):
    ax.scatter(x[clusters==i,0], x[clusters==i,1],x[clusters==i,2],
                color=colors[i], label='Cluster ' + str(i + 1))

#Plotting the centroids of the clusters
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2],s=100, c='lightskyblue', label='Centroids')
plt.legend()
plt.title('K-means Clustering')
ax.set_zlabel(dataset.columns[0]) #label
ax.set_ylabel(dataset.columns[1])
ax.set_xlabel(dataset.columns[2])
plt.show()


#Part 2: Find the optimum number of clusters for k-means
from sklearn.cluster import KMeans

wcss = []

# Trying kmeans for k=1 to k=10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # within cluster sum of squares
plt.show()


# Part 3: Actual Categorization
ax=plt.subplot(111,projection='3d')
quality = np.reshape(dataset.loc[:, ['quality']].values, (-1,))
i = 0
for label in np.unique(quality):
    ax.scatter(x[quality==label,0], x[quality==label,1],x[quality==label,2],
                color=colors[i], label=label)
    i += 1

plt.legend()
plt.title('Quality')
ax.set_xlabel(dataset.columns[0])
ax.set_ylabel(dataset.columns[1])
ax.set_ylabel(dataset.columns[2])
plt.show()