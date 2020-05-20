import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from keras.layers.core import Dense, Activation
from keras.models import Sequential

def ConvertToDict(df,df_column_name):
    df_np_array = df[df_column_name].unique()
    # print(type(df_np_array), df_np_array)
    df_column_dict = {}
    counter = 0
    for df_np_array_data in df_np_array:
        df_column_dict[df_np_array_data] = counter
        counter += 1
    return df_column_dict

csvfile = "winequalityN.csv"
#read the csv
df_wine = pd.read_csv(csvfile)
# print(df_wine.count())
#drop rows that have missing data.
df_wine.dropna(inplace = True)
df_wine.to_csv("temp.csv")

#print the new sample count
print(df_wine.count())

#print the column names
print(df_wine.columns)

#create dictionary
wine_type_dict = ConvertToDict(df_wine,"type")
# room_type_dict = ConvertToDict(df_wine,"room_type")
#
df_wine["type"].replace(wine_type_dict,inplace = True)

#plot correlation matrix
corr_mat = df_wine.corr()
plt.figure(figsize=(15,5))
sns.heatmap(data=corr_mat, annot=True, cmap='GnBu')
plt.show()

target = 'quality'
candidates = corr_mat.index[
    (corr_mat[target] > 0.4) | (corr_mat[target] < -0.4)
].values
candidates = candidates[candidates != target]
print('Correlated to', target, ': ', candidates)

def CustomizedLogReg(df,x_train_column,y_train_column):
    # using logistic regression to determine area [max_iter if set to anything less than 150 it will not converge]
    logReg = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42, C=1e3, max_iter=200)

    # Comparison 1 - Lat,long against neighbourhood
    x = df[x_train_column]
    y = df[y_train_column]

    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40, test_size=0.2)
    # display count of each data
    print("display count values")
    print(x_train.count(), y_train.count(), x_test.count(), y_test.count())

    # Model Fitting for Comparison 1 and Comparison 2
    logReg.fit(x_train, y_train)

    # Prediction for Comparison 1 and Comparison 2
    y_pred = logReg.predict(x_test)

    # print("--")
    # print("y_test")
    # print(y_test)
    # print("y_pred")
    # print(y_pred.tolist())

    # Check Accuracy Score for Comparison 1 and Comparison 2
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy of Comparison: ", accuracy)

    # Produce Confusion Matrices
    # cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    # print("Confusion Matrix")
    # print(cm)

"""
Main
"""
# x_train_column = ["type","fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
x_train_column = ["alcohol"]
y_train_column = "quality"
CustomizedLogReg(df_wine,x_train_column,y_train_column)


"""
Neural Network
"""
# df_wine here have dropna before
# df_train = df_wine.sample(frac = 0.25)
# df_test = df_wine[~df_wine.isin(df_train)]
# print(df_test)
print("-------------------------------------")
print("Neural Network")
print("-------------------------------------")

def CreateDF_test_train_for_Neural_Network(df,fraction = 0.9):
    df_train = df.sample(frac=fraction)
    df_test = df[~df.isin(df_train)]
    df_train.dropna(axis=0, inplace=True)
    df_test.dropna(axis=0, inplace=True)
    return df_train,df_test
#
df_train,df_test = CreateDF_test_train_for_Neural_Network(df_wine)

x_columns = [11]

y_columns = [12]

#
df_test.to_csv("test.csv")
df_train.to_csv("train.csv")

def CreateNP_array_for_Neural_Network(df,x_columns,y_columns):

    X = df.iloc[:,x_columns].to_numpy()
    Y = df.iloc[:,y_columns].to_numpy()
    return X,Y
#
X_train,Y_train = CreateNP_array_for_Neural_Network(df_train,x_columns,y_columns)
X_test,Y_test = CreateNP_array_for_Neural_Network(df_test,x_columns,y_columns)
# print(X_test)
# print(Y_test)

def NeuralNetworkModel1(X_train,Y_train):
    model = Sequential()
    model.add(Dense(200, input_shape=(11,), activation='sigmoid'))
    # model.add(Dense(2000, activation='relu'))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])
    model.fit(X_train, Y_train, batch_size = 64, epochs=10, verbose=1)
    return model
# model = NeuralNetworkModel1(X_train, Y_train)
# loss = model.evaluate(X_test, Y_test, verbose=1)
# # print('Loss = ', loss )
# predictions = model.predict(X_test)
# for i in np.arange(len(predictions)):
#     # print('Data: ', X_test[i], ', Actual: ', Y_test[i], ', Predicted: ', predictions[i])
#     print(',Actual: ', Y_test[i], ', Predicted: ', predictions[i])

def NeuralNetworkModel2(X_train,Y_train,X_test,Y_test,min_dense_value = 200, max_dense_value = 500,increment = 100):
    df = pd.DataFrame(columns=("dense value", "loss", "duration", "Y_test", "Y_pred"))
    counter = 0
    for j in range(min_dense_value,max_dense_value,increment):
        start_time = time.time()
        model = Sequential()
        model.add(Dense(j, input_shape=(11,), activation='sigmoid'))
        # model.add(Dense(j, activation='relu'))
        # model.add(Dense(j, activation='tanh'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])
        model.fit(X_train, Y_train, batch_size = 64, epochs=100, verbose=1)
        model = NeuralNetworkModel1(X_train, Y_train)
        loss = model.evaluate(X_test, Y_test, verbose=1)
        print('Loss = ', loss )
        predictions = model.predict(X_test)
        for i in np.arange(len(predictions)):
            # print('Data: ', X_test[i], ', Actual: ', Y_test[i], ', Predicted: ', predictions[i])
            Y_test_list = Y_test[i].tolist()
            predictions_list = predictions[i]
            print(',Actual: ', Y_test_list[0], ', Predicted: ', predictions_list[0])
            duration = (time.time() - start_time)
            df.loc[counter] = [j,loss, duration, Y_test_list[0],predictions_list[0]]
            counter += 1
        duration = (time.time() - start_time)
        print("--- %s seconds ---" % (duration))
    df.to_csv("1-sigmoid 100-epoch.csv")

# NeuralNetworkModel2(X_train,Y_train,X_test,Y_test,500,2100,500)

def NeuralNetworkModel3(X_train,Y_train,X_test,Y_test,min_dense_value = 200, max_dense_value = 500,increment = 100):
    df = pd.DataFrame(columns=("dense value", "loss", "duration", "Y_test", "Y_pred"))
    counter = 0
    for j in range(min_dense_value,max_dense_value,increment):
        start_time = time.time()
        model = Sequential()
        model.add(Dense(j, input_shape=(11,), activation='tanh'))
        # model.add(Dense(j, activation='sigmoid'))
        # model.add(Dense(j, activation='tanh'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])
        model.fit(X_train, Y_train, batch_size = 64, epochs=100, verbose=1)
        model = NeuralNetworkModel1(X_train, Y_train)
        loss = model.evaluate(X_test, Y_test, verbose=1)
        print('Loss = ', loss )
        predictions = model.predict(X_test)
        for i in np.arange(len(predictions)):
            # print('Data: ', X_test[i], ', Actual: ', Y_test[i], ', Predicted: ', predictions[i])
            Y_test_list = Y_test[i].tolist()
            predictions_list = predictions[i]
            print(j,',Actual: ', Y_test_list[0], ', Predicted: ', predictions_list[0])
            duration = (time.time() - start_time)
            df.loc[counter] = [j,loss, duration, Y_test_list[0],predictions_list[0]]
            counter += 1
        duration = (time.time() - start_time)
        print("--- %s seconds ---" % (duration))
    df.to_csv("1-tanh 100-epoch.csv")

# NeuralNetworkModel3(X_train,Y_train,X_test,Y_test,500,2100,500)

def NeuralNetworkModel4(X_train,Y_train,X_test,Y_test,min_dense_value = 200, max_dense_value = 500,increment = 100):
    df = pd.DataFrame(columns=("dense value", "loss", "duration", "Y_test", "Y_pred"))
    counter = 0
    for j in range(min_dense_value,max_dense_value,increment):
        start_time = time.time()
        model = Sequential()
        model.add(Dense(j, input_shape=(11,), activation='sigmoid'))
        model.add(Dense(j, activation='sigmoid'))
        # model.add(Dense(j, activation='tanh'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])
        model.fit(X_train, Y_train, batch_size = 64, epochs=100, verbose=1)
        model = NeuralNetworkModel1(X_train, Y_train)
        loss = model.evaluate(X_test, Y_test, verbose=1)
        print('Loss = ', loss )
        predictions = model.predict(X_test)
        for i in np.arange(len(predictions)):
            # print('Data: ', X_test[i], ', Actual: ', Y_test[i], ', Predicted: ', predictions[i])
            Y_test_list = Y_test[i].tolist()
            predictions_list = predictions[i]
            print(j,',Actual: ', Y_test_list[0], ', Predicted: ', predictions_list[0])
            duration = (time.time() - start_time)
            df.loc[counter] = [j,loss, duration, Y_test_list[0],predictions_list[0]]
            counter += 1
        duration = (time.time() - start_time)
        print("--- %s seconds ---" % (duration))
    df.to_csv("2 sigmoid 100-epoch.csv")

# NeuralNetworkModel4(X_train,Y_train,X_test,Y_test,500,2100,500)

def NeuralNetworkModel5(X_train,Y_train,X_test,Y_test,min_dense_value = 200, max_dense_value = 500,increment = 100):
    df = pd.DataFrame(columns=("dense value", "loss", "duration", "Y_test", "Y_pred"))
    counter = 0
    for j in range(min_dense_value,max_dense_value,increment):
        start_time = time.time()
        model = Sequential()
        model.add(Dense(j, input_shape=(11,), activation='sigmoid'))
        model.add(Dense(j, activation='tanh'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])
        model.fit(X_train, Y_train, batch_size = 64, epochs=100, verbose=1)
        model = NeuralNetworkModel1(X_train, Y_train)
        loss = model.evaluate(X_test, Y_test, verbose=1)
        print('Loss = ', loss )
        predictions = model.predict(X_test)
        for i in np.arange(len(predictions)):
            # print('Data: ', X_test[i], ', Actual: ', Y_test[i], ', Predicted: ', predictions[i])
            Y_test_list = Y_test[i].tolist()
            predictions_list = predictions[i]
            print(j,',Actual: ', Y_test_list[0], ', Predicted: ', predictions_list[0])
            duration = (time.time() - start_time)
            df.loc[counter] = [j,loss, duration, Y_test_list[0],predictions_list[0]]
            counter += 1
        duration = (time.time() - start_time)
        print("--- %s seconds ---" % (duration))
    df.to_csv("1-sigmoid 1-tanh 100-epoch.csv")

# NeuralNetworkModel5(X_train,Y_train,X_test,Y_test,500,2100,200)

def NeuralNetworkModel6(X_train,Y_train,X_test,Y_test,min_dense_value = 200, max_dense_value = 500,increment = 100):
    df = pd.DataFrame(columns=("dense value", "loss", "duration", "Y_test", "Y_pred"))
    counter = 0
    for j in range(min_dense_value,max_dense_value,increment):
        start_time = time.time()
        model = Sequential()
        model.add(Dense(j, input_shape=(11,), activation='sigmoid'))
        model.add(Dense(j, activation='relu'))
        model.add(Dense(j, activation='tanh'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])
        model.fit(X_train, Y_train, batch_size = 64, epochs=100, verbose=1)
        model = NeuralNetworkModel1(X_train, Y_train)
        loss = model.evaluate(X_test, Y_test, verbose=1)
        print('Loss = ', loss )
        predictions = model.predict(X_test)
        for i in np.arange(len(predictions)):
            # print('Data: ', X_test[i], ', Actual: ', Y_test[i], ', Predicted: ', predictions[i])
            Y_test_list = Y_test[i].tolist()
            predictions_list = predictions[i]
            print(j,',Actual: ', Y_test_list[0], ', Predicted: ', predictions_list[0])
            duration = (time.time() - start_time)
            df.loc[counter] = [j,loss, duration, Y_test_list[0],predictions_list[0]]
            counter += 1
        duration = (time.time() - start_time)
        print("--- %s seconds ---" % (duration))
    df.to_csv("1-sigmoid 1-relu 1-tanh.csv")

# NeuralNetworkModel6(X_train,Y_train,X_test,Y_test,500,2100,500)

"""
EMPLOY PCA with LOGREG
"""

print("-------------------------------------")
print("PCA with LOGREG")
print("-------------------------------------")
y_pca = df_wine.iloc[:,y_columns].values
x_pca = StandardScaler().fit_transform(df_wine.iloc[:,x_columns])

pca = PCA(n_components=5)
pc = pca.fit_transform(x_pca)
# print(pc)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

def CustomizedLogRegWithPCA(x_pca,y_pca):
    # using logistic regression to determine area [max_iter if set to anything less than 150 it will not converge]
    logReg = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42, max_iter=500)
    # Comparison 1 - Lat,long against neighbourhood
    # x = df[x_train_column]
    # y = df[y_train_column]
    # split data
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y_pca, random_state=3, train_size=0.5)
    # display count of each data
    print("display count values")
    # print(x_train.count(), y_train.count(), x_test.count(), y_test.count())

    # Model Fitting for Comparison 1 and Comparison 2
    logReg.fit(x_train, y_train)

    # Prediction for Comparison 1 and Comparison 2
    y_pred = logReg.predict(x_test)

    print("--")
    print("y_test")
    print(y_test)
    print("y_pred")
    print(y_pred.tolist())

    # Check Accuracy Score for Comparison 1 and Comparison 2
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy of Comparison: ", accuracy)

CustomizedLogRegWithPCA(x_pca,y_pca)

