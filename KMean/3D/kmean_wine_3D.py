import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import datetime

#Part 1
starttime1 = datetime.datetime.now()

dataset = pd.read_csv('winequalityN.csv')
x = dataset.iloc[0:2000, [6,7,8]].values

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
endtime1 = datetime.datetime.now()
plt.title('K-means Clustering'+'__RunTime: '+str(endtime1 - starttime1))
ax.set_xlabel(dataset.columns[6]) #label
ax.set_ylabel(dataset.columns[7])
ax.set_zlabel(dataset.columns[8])
plt.show()


#Part 2: Find the optimum number of clusters for k-means
from sklearn.cluster import KMeans
starttime2 = datetime.datetime.now()
wcss = []

# Trying kmeans for k=1 to k=10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
endtime2 = datetime.datetime.now()
plt.title('The elbow method'+'__RunTime: '+str(endtime2 - starttime2))
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # within cluster sum of squares
plt.show()


# Part 3: Actual Categorization
starttime3 = datetime.datetime.now()
ax=plt.subplot(111,projection='3d')
quality = np.reshape(dataset.loc[0:1999, ['quality']].values, (-1,))
i = 0
for label in np.unique(quality):
    ax.scatter(x[quality==label,0], x[quality==label,1],x[quality==label,2],
                color=colors[i], label=label)
    i += 1

plt.legend()
endtime3 = datetime.datetime.now()
plt.title('Quality'+'__RunTime: '+str(endtime3 - starttime3))
ax.set_xlabel(dataset.columns[6])
ax.set_ylabel(dataset.columns[7])
ax.set_zlabel(dataset.columns[8])
plt.show()


# Part 4: Actual Categorization in range
starttime4 = datetime.datetime.now()
ax=plt.subplot(111,projection='3d')

dataset.loc[dataset['quality']==3,'quality_range']=1
dataset.loc[dataset['quality']==4,'quality_range']=2
dataset.loc[dataset['quality']==5,'quality_range']=2
dataset.loc[dataset['quality']==6,'quality_range']=2
dataset.loc[dataset['quality']==7,'quality_range']=3
dataset.loc[dataset['quality']==8,'quality_range']=3
dataset.loc[dataset['quality']==9,'quality_range']=3


quality = np.reshape(dataset.loc[0:1999, ['quality_range']].values, (-1,))
i = 0
for label in np.unique(quality):
    ax.scatter(x[quality==label,0], x[quality==label,1],x[quality==label,2],
                color=colors[i], label=label)
    i += 1

plt.legend()
endtime4 = datetime.datetime.now()
plt.title('Quality'+'__RunTime: '+str(endtime4 - starttime4))
ax.set_xlabel(dataset.columns[6])
ax.set_ylabel(dataset.columns[7])
ax.set_zlabel(dataset.columns[8])
plt.show()