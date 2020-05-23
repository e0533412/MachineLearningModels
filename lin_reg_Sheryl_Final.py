import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('kc_house_data.csv')
df.head()

pd.DataFrame(df.isna().sum()).T
df.info()

df.drop(["id", "date"], axis = 1, inplace = True)
df.head()

#check correlation between features
corr_mat = df.corr()

plt.figure(figsize=(15, 9))
sns.heatmap(data=corr_mat, annot=True, cmap='GnBu')
plt.show()

#Linear Regression

#Bivariate Linear Regression

#independent variable
X = df[['sqft_living']]

#dependent variable
Y = df['price']

#Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)

#instantiation of Linear Regression
simple_lr = LinearRegression()

#Timing the model training time
start_time = time.time()

#Model Training
simple_lr.fit(X_train,Y_train)

end_time = time.time()
print("-----%s seconds -----" % (end_time - start_time))

#Predicting the data
pred = simple_lr.predict(X_test)
print("Train Accuracy")
print(simple_lr.score(X_train,Y_train))
print("Test Accuracy")
print(simple_lr.score(X_test,Y_test))

#Model Validation
mean_squared_error(Y_test, pred)
r2_score(Y_test, pred)

#Multivariate Linear Regression

#Model 2
#independent variable
x_mod2 = df[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']]

#dependent variable
y_mod2 = df['price']

#Split the data
x_train, x_test, y_train, y_test = train_test_split(x_mod2, y_mod2, test_size=0.3, random_state=40)

#instantiation of Linear Regression
simple_lr2 = LinearRegression()

#Timing the model training time
start_time = time.time()

#Model Training
simple_lr2.fit(X_train,Y_train)

end_time = time.time()
print("-----%s seconds -----" % (end_time - start_time))

#Predicting the data
pred2 = simple_lr2.predict(x_test)

print("Train Accuracy")
print(simple_lr2.score(x_train,y_train))
print("Test Accuracy")
print(simple_lr2.score(x_test,y_test))

r2_score(y_test, pred2)
mean_squared_error(y_test, pred2)

#Model 3

column=list(df.columns)
feat = list(set(column)-set(["price"]))

price = df["price"].values
house = df[feat].values

# price = np.log(price) #run natural log for interpretation - change in natural log = percentage change

# #Pre-processing
# from sklearn.preprocessing import StandardScaler
# sn = StandardScaler();
# house=sn.fit_transform(house)

#Split the data
xtrain, xtest, ytrain, ytest = train_test_split(house, price, test_size=0.3, random_state=40)

#instantiation of Linear Regression
lr = LinearRegression(fit_intercept = True)

#Timing the model training time
start_time = time.time()

#Model Training
model =lr.fit(xtrain, ytrain)

end_time = time.time()
print("-----%s seconds -----" % (end_time - start_time))

#Predicting the model
pred3 = lr.predict(xtest)

print("Train Accuracy")
print(lr.score(xtrain,ytrain))
print("Test Accuracy")
print(lr.score(xtest,ytest))

r2_score(ytest, pred3)
mean_squared_error(ytest, pred3)
