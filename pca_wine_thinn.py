import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv('../data/winequalityN.csv')
# fill NaN with 0.0
df_wine.fillna(0.0, inplace=True)

#convert type to 0,1
def convertToNum(x):
    if x == "white":
        return 0
    else:
        return 1

df_wine["type"] = df_wine["type"].apply(convertToNum)

y = df_wine.loc[:, 'type'].values
x = StandardScaler().fit_transform(df_wine.iloc[:, 1:13])

pca = PCA(n_components=5)
pc = pca.fit_transform(x)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

# 2Dplot
colors = 'rgbkcmy'

for name in np.unique(y):
    plt.scatter(pc[y==name, 0], pc[y==name, 1],
                color=colors[name],
                label =name)

plt.legend()
plt.title('After PCA Transformation')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()