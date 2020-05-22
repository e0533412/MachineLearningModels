import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
import datetime

starttime=datetime.datetime.now()
dataset = pd.read_csv('winequalityN.csv')
x = dataset.iloc[0:2000, [6,7,8]].values

dbscan = DBSCAN(eps=5, min_samples = 5)
clusters = dbscan.fit_predict(x)

colors = 'rgbkcmy'

ax=plt.subplot(111,projection='3d')

for i in np.unique(clusters):
    label = 'Outlier' if i == -1 else 'Cluster ' + str(i + 1)
    ax.scatter(x[clusters==i,0], x[clusters==i,1],x[clusters==i,2],
               color=colors[i], label=label)

plt.legend()
endtime=datetime.datetime.now()
plt.title('DBSCAN_WINE_RunTime: '+str(endtime-starttime))
ax.set_xlabel(dataset.columns[6])
ax.set_ylabel(dataset.columns[7])
ax.set_zlabel(dataset.columns[8])
plt.show()