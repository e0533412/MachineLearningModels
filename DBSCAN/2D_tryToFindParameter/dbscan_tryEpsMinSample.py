import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import datetime
import math
import operator

#try to find eps, min_samples

dataset = pd.read_csv('winequalityN.csv')
x1 = dataset.iloc[0:200, [6,7]].values

def euclideanDistance(X,Y):
    return np.sqrt(np.sum(np.square(X-Y)))

k = 4
distances = {}
for x in range(len(x1)):
    distancesForRow = []
    for y in range(len(x1)):
        if x != y:
            dist = euclideanDistance(x1[x],x1[y])
            distancesForRow.append(dist)
    distancesForRow.sort(reverse = True)
    distances[x] = distancesForRow[k-1]

newDistances = sorted(distances.items(), key = operator.itemgetter(1))
for x in range(len(newDistances)):
    plt.scatter(x, newDistances[x][1])

plt.show()

