import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

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

x = df_bins.iloc[:, :14].values

#Applying dbscan to the dataset / receiving
# cluster-classification for every row
dbscan = DBSCAN(eps=5, min_samples = 50)
clusters = dbscan.fit_predict(x)

colors = [
    'red', 'blue', 'green', 'chocolate',
    'lightgreen', 'cornflowerblue',
    'violet', 'darkturquoise', 'lightskyblue',
    'magenta', 'yellowgreen', 'coral',
    'darkorange', 'mediumpurple',
    'olive', 'tan',
]

#print out countries under each cluster
for i in np.unique(clusters):
    label = 'Outlier' if i == -1 else 'Cluster ' + str(i + 1)
    c = df_bins.iloc[clusters==i, 0]
    print(label + ': ', c.values, '\n')
#plot cluster
fig = plt.figure(figsize=(7, 7))
#2D plot
colors = 'rgbkcmy'

for i in np.unique(clusters):
    label = 'Outlier' if i == -1 else 'Cluster ' + str(i + 1)
    plt.scatter(x[clusters==i,1], x[clusters==i,11],
                color=colors[i], label=label)

plt.xlabel(df_bins.columns[1])
plt.ylabel(df_bins.columns[11])
plt.legend()
plt.show()

