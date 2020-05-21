import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import axes3d

dataset = pd.read_csv('data/winequalityN.csv')
dataset = dataset.fillna(0)

y = dataset.loc[:,'quality'].values
x = StandardScaler().fit_transform(dataset.iloc[:,1:11])

pca = PCA(n_components=3)
pc = pca.fit_transform(x)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

colors = 'rgbmykc'

fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection='3d')

for i in np.unique(y):
    ax.scatter3D(pc[y==i,0], pc[y==i,1],
             pc[y==i,2], color=colors[i],
               label='quality ' + str(i))

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")

plt.legend()
plt.title('PCA on wine dataset')
plt.show()
