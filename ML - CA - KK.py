import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



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
df_sg_listing = pd.read_csv(csvfile)
print(df_sg_listing)

print(df_sg_listing.columns)

neighbourhood_dict = ConvertToDict(df_sg_listing,"neighbourhood")
neighbourhood_group_dict = ConvertToDict(df_sg_listing,"neighbourhood_group")

print(neighbourhood_dict)
print()
print(neighbourhood_group_dict)


# """
# Feature Engineering
# """
# # my target output is to check what area in terms of longtitude and latitude is within this neighbourhood
# target1 = "neighbourhood"
#
# corr_mat = df_sg_listing.corr()
# plt.figure(figsize = (15,5))
# sns.heatmap(data=corr_mat,annot=True,cmap = 'GnBu')
# plt.show()
#
