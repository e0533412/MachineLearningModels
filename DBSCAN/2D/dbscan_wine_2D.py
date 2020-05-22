import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import datetime

starttime=datetime.datetime.now()
dataset = pd.read_csv('winequalityN.csv')
x = dataset.iloc[0:2000, [6,7]].values

dbscan = DBSCAN(eps=5, min_samples = 5)
clusters = dbscan.fit_predict(x)

colors = 'rgbkcmy'

for i in np.unique(clusters):
    label = 'Outlier' if i == -1 else 'Cluster ' + str(i + 1)
    plt.scatter(x[clusters==i,0], x[clusters==i,1],
               color=colors[i], label=label)

plt.legend()
endtime=datetime.datetime.now()
plt.title('DBSCAN_WINE_RunTime: '+str(endtime-starttime))
plt.show()


