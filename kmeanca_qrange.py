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

#binning
df_bins = df.copy()
print(df_bins.shape)
bins = [0,4,7,10]
labels = [0,1,2]
df_bins['quality_range'] = pd.cut(x = df_bins['quality'], bins = bins, labels = labels)

x = df_bins.iloc[:, :13].values

colors = 'rgbkcmy'

#Find the optimum number of clusters for k-means
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

#Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3)
clusters = kmeans.fit_predict(x)

#3Dplot
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection='3d')

for i in np.unique(clusters):
    ax.scatter3D(x[clusters==i,1],
             x[clusters==i,3],
             x[clusters==i,11],
                 color=colors[i], label='Cluster ' + str(i + 1))

ax.set_xlabel(df_bins.columns[1])
ax.set_ylabel(df_bins.columns[3])
ax.set_zlabel(df_bins.columns[11])

plt.legend()
plt.title('K-Means Clustering')
plt.show()

#2D
for i in np.unique(clusters):
    label = 'Outlier' if i == -1 else 'Cluster ' + str(i + 1)
    plt.scatter(x[clusters==i,1], x[clusters==i,11],
                color=colors[i], label=label)

plt.xlabel(df_bins.columns[1])
plt.ylabel(df_bins.columns[11])
plt.legend()
plt.show()
