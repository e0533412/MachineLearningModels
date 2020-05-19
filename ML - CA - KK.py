import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



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
print(df_sg_listing.count())
#drop rows that have missing data.
df_sg_listing.dropna(inplace = True)

#print the new sample count
print(df_sg_listing.count())

#print the column names
print(df_sg_listing.columns)

#create dictionary
neighbourhood_dict = ConvertToDict(df_sg_listing,"neighbourhood")
neighbourhood_group_dict = ConvertToDict(df_sg_listing,"neighbourhood_group")

#see the dictionary
print(neighbourhood_dict)
print()
print(neighbourhood_group_dict)

#replace values in the dataframe with integers
df_sg_listing["neighbourhood"].replace(neighbourhood_dict,inplace = True)
df_sg_listing["neighbourhood_group"].replace(neighbourhood_group_dict,inplace = True)

#print df to see
print(df_sg_listing)

#using logistic regression to determine area
logReg = LogisticRegression(solver = 'lbfgs')

x1 = df_sg_listing[["latitude","longitude"]]
y1 = df_sg_listing["neighbourhood"]

x2 = df_sg_listing[["latitude","longitude"]]
y2 = df_sg_listing["neighbourhood_group"]

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state = 1)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state = 1)

print(x1_train.count(),y1_train.count(), x1_test.count(), y1_test.count())
print()
print(x2_train.count(),y2_train.count(), x2_test.count(), y2_test.count())




"""
Feature Engineering
"""
# my target output is to check what area in terms of longtitude and latitude is within this neighbourhood
target1 = "neighbourhood"

corr_mat = df_sg_listing.corr()
plt.figure(figsize = (15,5))
sns.heatmap(data=corr_mat,annot=True,cmap = 'GnBu')
plt.show()

