import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('kc_house_data.csv')
df.dropna(inplace = True)
df.head()

df.drop(["id", "date"], axis = 1, inplace = True)

#Model Training
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

column=list(df.columns)
feat = list(set(column)-set(["price"]))

price = df["price"].values
house = df[feat].values

price = np.log(price) #run natural log for interpretation - change in natural log = percentage change

#pre-processing
from  sklearn.preprocessing import StandardScaler
sn= StandardScaler();
house=sn.fit_transform(house)

xtrain, xtest, ytrain, ytest = train_test_split(house, price, test_size=0.25, random_state=0)

#Model Testing
lr = LinearRegression(fit_intercept = True)
model =lr.fit(xtrain, ytrain)

prediction=lr.predict(xtest)

print("Train Accuracy")
print(lr.score(xtrain,ytrain))
print("Test Accuracy")
print(lr.score(xtest,ytest))
