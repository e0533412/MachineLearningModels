import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm
import time

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

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

def CreateDF_test_train_for_Neural_Network(df,fraction = 0.9):
    df_train = df.sample(frac=fraction)
    df_test = df[~df.isin(df_train)]
    df_train.dropna(axis=0, inplace=True)
    df_test.dropna(axis=0, inplace=True)
    return df_train,df_test

def CustomizedLogReg(df,x_columns,y_columns):
    # using logistic regression to determine area [max_iter if set to anything less than 150 it will not converge]

    # Comparison 1 - Lat,long against neighbourhood
    x = df.iloc[:,x_columns]
    y = df.iloc[:,y_columns]

    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

    print("----------------------------------")
    print("X_test.head")
    print("----------------------------------")
    print(x_test.head())

    ##Prepare Log Reg Model
    logReg = LogisticRegression(random_state=40)

    logReg.fit(x_train, y_train)

    train_accuracy = logReg.score(x_train, y_train)
    test_accuracy = logReg.score(x_test, y_test)
    print('One-vs-rest', '-' * 35,
          'Accuracy Score of Train Model : {:.2f}'.format(train_accuracy),
          'Accuracy Score of Test  Model : {:.2f}'.format(test_accuracy), sep='\n')

    y_pred = logReg.predict(x_test)
    score = round(accuracy_score(y_test, y_pred), 3)
    cm1 = cm(y_test, y_pred)
    sns.heatmap(cm1, annot=True, fmt=".0f")
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Accuracy Score: {0}'.format(score), size=15)
    plt.show()
    print(cm)


def CreateNP_array_for_Neural_Network(df,x_columns,y_columns):

    X = df.iloc[:,x_columns].to_numpy()
    Y = df.iloc[:,y_columns].to_numpy()
    return X,Y


def NeuralNetworkModel1(X_train,Y_train):
    model = Sequential()
    model.add(Dense(200, input_shape=(X_train.shape[1],), activation='sigmoid'))
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
        model = Sequential()
        model.add(Dense(j, input_shape=(X_train.shape[1],), activation='sigmoid'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])

        ### CHECK HOW LONG NEEDED TO DO MODEL FITTING
        start_time = time.time()
        model.fit(X_train, Y_train, batch_size = 64, epochs=100, verbose=1)
        duration = (time.time() - start_time)
        print("--- %s seconds ---" % (duration))

        ### CHECK ON LOSS VALUE
        loss = model.evaluate(X_test, Y_test, verbose=1)
        print('Loss = ', loss )
        predictions = model.predict(X_test)
        for i in np.arange(len(predictions)):
            Y_test_list = Y_test[i].tolist()
            predictions_list = predictions[i]
            print(',Actual: ', Y_test_list[0], ', Predicted: ', predictions_list[0])
            duration = (time.time() - start_time)
            df.loc[counter] = [j,loss, duration, Y_test_list[0],predictions_list[0]]
            counter += 1
    df.to_csv("1-sigmoid x = 1,2,3,4,5,6,7,8,9,10,11 y = 13.csv")

def NeuralNetworkModel6(X_train,Y_train,X_test,Y_test,min_dense_value = 200, max_dense_value = 500,increment = 100):
    df = pd.DataFrame(columns=("dense value", "loss", "duration", "Y_test", "Y_pred"))
    counter = 0
    for j in range(min_dense_value,max_dense_value,increment):
        start_time = time.time()
        model = Sequential()
        model.add(Dense(j, input_shape=(X_train.shape[1],), activation='sigmoid'))
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

def CustomizedLogRegWithPCA(x_pca,y_pca):
    # using logistic regression to determine area [max_iter if set to anything less than 150 it will not converge]
    logReg = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42, max_iter=500)
    # Comparison 1 - Lat,long against neighbourhood
    # x = df[x_train_column]
    # y = df[y_train_column]
    # split data
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y_pca, random_state=3, train_size=0.7)
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


def GetSumAndPercentageOfNullValues(df):
    total_missing_values = df.isnull().sum()
    missing_values_per = df.isnull().sum()/df.isnull().count()
    null_values = pd.concat([total_missing_values, missing_values_per], axis=1, keys=['total_null', 'total_null_perc'])
    null_values = null_values.sort_values('total_null', ascending=False)
    print(null_values[null_values['total_null'] > 0])
    return null_values[null_values['total_null'] > 0]

"""
Main
"""
csvfile = "winequalityN.csv"
#read the csv
df_wine = pd.read_csv(csvfile)
# print(df_wine.count())
sns.heatmap(df_wine.isnull(),yticklabels=False,cbar=False,cmap='viridis')

###Check with Null values
Sum = df_wine.isnull().sum()
Percentage = ( df_wine.isnull().sum()/df_wine.isnull().count())

fill_list = (GetSumAndPercentageOfNullValues(df_wine)).index
print(type(fill_list),fill_list)

print("----------------------------------")
print("Dataframe Information before dropping NA")
print("----------------------------------")

print(df_wine.info())

#drop data with NaN values
df_wine.dropna(inplace = True)

print("----------------------------------")
print("Dataframe Information after dropping NA")
print("----------------------------------")

print(df_wine.info())
sns.heatmap(df_wine.isnull(),yticklabels=False,cbar=False,cmap='viridis')

### Show the correlation matrix
corr_matrix = df_wine.corr()
corr_list = corr_matrix.quality.abs().sort_values(ascending=False).index[0:]

print("----------------------------------")
print("Correlation List")
print("----------------------------------")
print(corr_list)
plt.figure(figsize=(11,9))
dropSelf = np.zeros_like(corr_matrix)
dropSelf[np.triu_indices_from(dropSelf)] = True
sns.heatmap(corr_matrix, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f", mask=dropSelf)
sns.set(font_scale=1.5)

print("----------------------------------")
print("Creating 2 Bins Model of Two Types of Wine Quality Classes")
print("----------------------------------")
df_bin = df_wine.copy()
wine_type_dict = ConvertToDict(df_bin,"type")
df_bin["type"].replace(wine_type_dict,inplace = True)
bins = [0,4,7,10]
labels = [0,1,2]
df_bin['quality_range']= pd.cut(x=df_bin['quality'], bins=bins, labels=labels)
print(df_bin[['quality_range','quality']].head())
print(df_bin.head())

#ASSIGN COLUMNS YOU WANT TO USE BASED ON df_bin
x_columns = [0,1,2,3,4,5,6,7,8,9,10,11]
y_columns = [13]

CustomizedLogReg(df_bin,x_columns,y_columns)

"""
Neural Network
"""
# df_wine here have dropna before
print("-------------------------------------")
print("Neural Network")
print("-------------------------------------")

df_train,df_test = CreateDF_test_train_for_Neural_Network(df_bin)

X_train,Y_train = CreateNP_array_for_Neural_Network(df_train,x_columns,y_columns)
X_test,Y_test = CreateNP_array_for_Neural_Network(df_test,x_columns,y_columns)

NeuralNetworkModel2(X_train,Y_train,X_test,Y_test,500,2100,500)
#
# # NeuralNetworkModel3(X_train,Y_train,X_test,Y_test,500,2100,500)
#
# # NeuralNetworkModel4(X_train,Y_train,X_test,Y_test,500,2100,500)
#
# # NeuralNetworkModel5(X_train,Y_train,X_test,Y_test,500,2100,200)
#
# # NeuralNetworkModel6(X_train,Y_train,X_test,Y_test,500,2100,500)
#
# # NeuralNetworkModel2(X_train,Y_train,X_test,Y_test,500,2100,500) #accuracy around 0.49
#
# # NeuralNetworkModel3(X_train,Y_train,X_test,Y_test,500,1100,500) #accuracy around 0.49
#
# """
# EMPLOY LOGREG
# """
#
# print("-------------------------------------")
# print("LOGREG")
# print("-------------------------------------")
#
# CustomizedLogReg(df_wine,x_columns,y_columns)
#
#
# """
# EMPLOY PCA with LOGREG
# """
#
# print("-------------------------------------")
# print("PCA with LOGREG")
# print("-------------------------------------")
# y_pca = df_wine.iloc[:,y_columns].values
# x_pca = StandardScaler().fit_transform(df_wine.iloc[:,x_columns])
#
# pca = PCA(n_components=1)
# pc = pca.fit_transform(x_pca)
# # print(pc)
#
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_ratio_.sum())
#
#
# CustomizedLogRegWithPCA(x_pca,y_pca)
#
#
#
#
