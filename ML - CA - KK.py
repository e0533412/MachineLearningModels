import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def ConvertToDict(df,df_column_name):
    df_np_array = df[df_column_name].unique()
    # print(type(df_np_array), df_np_array)
    df_column_dict = {}
    counter = 0
    for df_np_array_data in df_np_array:
        df_column_dict[df_np_array_data] = counter
        counter += 1
    return df_column_dict

csvfile = "SG_listings.csv"
#read the csv
df_sg_listing = pd.read_csv(csvfile)
# print(df_sg_listing.count())
#drop rows that have missing data.
# df_sg_listing.dropna(inplace = True)
# df_sg_listing.to_csv("temp.csv")

#print the new sample count
print(df_sg_listing.count())

#print the column names
print(df_sg_listing.columns)

#create dictionary
neighbourhood_dict = ConvertToDict(df_sg_listing,"neighbourhood")
neighbourhood_group_dict = ConvertToDict(df_sg_listing,"neighbourhood_group")
df_sg_listing["neighbourhood"].replace(neighbourhood_dict,inplace = True)
df_sg_listing["neighbourhood_group"].replace(neighbourhood_group_dict,inplace = True)


# #see the dictionary
# print(neighbourhood_dict)
# print()
# print(neighbourhood_group_dict)
#
# #replace values in the dataframe with integers
# df_sg_listing["neighbourhood"].replace(neighbourhood_dict,inplace = True)
# df_sg_listing["neighbourhood_group"].replace(neighbourhood_group_dict,inplace = True)
#
# #print df to see
# print(df_sg_listing["neighbourhood"].head(),df_sg_listing["neighbourhood_group"].head())
#
# #using logistic regression to determine area [max_iter if set to anything less than 150 it will not converge]
# logReg1 = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', random_state = 42,C=1e9, max_iter = 500)
# logReg2 = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', random_state = 42,C=1e9, max_iter = 500)
#
# #Comparison 1 - Lat,long against neighbourhood
# x1 = df_sg_listing[["latitude","longitude"]]
# y1 = df_sg_listing["neighbourhood"]
#
# #Comparison 2 - Lat,Long against neighbourhood group
# x2 = df_sg_listing.iloc[:,6:8]
# y2 = df_sg_listing["neighbourhood_group"]
#
# #plot Comparison 2
# df_1 = df_sg_listing["latitude"]
# df_2 = df_sg_listing["longitude"]
# df_3 = df_sg_listing["neighbourhood"]
# df_4 = df_sg_listing["neighbourhood_group"]
# df_test1 = pd.concat([df_1,df_2,df_3], axis = 1)
# df_test2 = pd.concat([df_1,df_2,df_4], axis = 1)
#
# df_test2
# print("df_test1")
# print(df_test1.head())
# print("df_test2")
# print(df_test2.head())
#
# # sns.pairplot(df_test1, hue='neighbourhood')  # Show different levels of a categorical variable by the color of plot elements
# # plt.show()
#
# sns.pairplot(df_test2, hue='neighbourhood_group')  # Show different levels of a categorical variable by the color of plot elements
# plt.show()
#
#
#
# #split data
# x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state = 3,train_size= 0.5)
# x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state = 3,train_size= 0.5)
#
# #display count of each data
# print("display count values")
# print(x1_train.count(),y1_train.count(), x1_test.count(), y1_test.count())
# print()
# print(x2_train.count(),y2_train.count(), x2_test.count(), y2_test.count())
#
# #Model Fitting for Comparison 1 and Comparison 2
# logReg1.fit(x1_train, y1_train)
# logReg2.fit(x2_train, y2_train)
#
# #Prediction for Comparison 1 and Comparison 2
# y1_pred = logReg1.predict(x1_test)
# y2_pred = logReg2.predict(x2_test)
#
# print("--")
# print("y1_test")
# print(y1_test)
# print("y1_pred")
# print(y1_pred.tolist())
#
#
# print("-------------------------")
# print("y2_test")
# print(y2_test)
# print("y2_pred")
# print(y2_pred.tolist())
#
# #Check Accuracy Score for Comparison 1 and Comparison 2
# accuracy1 = accuracy_score(y1_test, y1_pred)
# accuracy2 = accuracy_score(y2_test, y2_pred)
#
# print("Accuracy of Comparison 1: " , accuracy1)
# print("Accuracy of Comparison 2: " , accuracy2)
#
# #Produce Confusion Matrices
# cm1 = confusion_matrix(y1_test, y1_pred, labels = [1,0])
# cm2 = confusion_matrix(y2_test, y2_pred, labels = [1,0])
# print("Confusion Matrix 1")
# print(cm1)
# print("Confusion Matrix 2")
# print(cm2)

def CustomizedLogReg(df,x_train_column,y_train_column):
    # using logistic regression to determine area [max_iter if set to anything less than 150 it will not converge]
    logReg = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42, C=1e9, max_iter=500)

    # Comparison 1 - Lat,long against neighbourhood
    x = df[x_train_column]
    y = df[y_train_column]

    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, train_size=0.5)
    # display count of each data
    print("display count values")
    print(x_train.count(), y_train.count(), x_test.count(), y_test.count())

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

    # Produce Confusion Matrices
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    print("Confusion Matrix")
    print(cm)

"""
Main
"""
x_train_column = ["latitude", "longitude"]
y_train_column = "neighbourhood_group"
CustomizedLogReg(df_sg_listing,x_train_column,y_train_column)

