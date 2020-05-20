import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from keras.layers.core import Dense, Activation
from keras.models import Sequential

winedata = pd.read_csv('winequalityN.csv')
winedata.dropna(inplace = True)

def ConvertToDict(winedata, type):
    winedata_np_array = winedata[type].unique()
    # print(type(df_np_array), df_np_array)
    df_column_dict = {}
    counter = 0
    for winedata_np_array_data in winedata_np_array:
        df_column_dict[winedata_np_array_data] = counter
        counter += 1
    return df_column_dict

#create dictionary
wine_type_dict = ConvertToDict(winedata,"type")
winedata["type"].replace(wine_type_dict,inplace = True)

# df_train = winedata.sample(frac=0.75)
# df_test = winedata[~winedata.isin(df_train)]
# df_train.dropna(axis=0, inplace=True)
# df_test.dropna(axis=0, inplace=True)

X_columns = [1,2,3,4,5,6,7,8,9,10,11]
Y_columns = [0]

print(winedata.iloc[:,X_columns])


X_train = np.array(winedata.iloc[:,X_columns])
Y_train = np.array(winedata.iloc[:,Y_columns])

X_test = np.array(winedata.iloc[:,X_columns])
Y_test = np.array(winedata.iloc[:,Y_columns])

# Y_train_ohe = np.array([[0] if winedata.type == 'white' else [1] for winedata.type in Y_train])

# X_train, X_test, Y_train, Y_test = train_test_split(
#     winedata.iloc[:, winedata.columns != 'type'], winedata['type'], test_size=0.25, random_state=42)

# X_train, X_test, Y_train, Y_test = train_test_split(winedata.iloc[:,X_columns],winedata.iloc[:,Y_columns], test_size=0.20, random_state=42)

#construct neural network model
model = Sequential()
model.add(Dense(500, input_shape=(X_train.shape[1],), activation='sigmoid'))
model.add(Dense(500, activation='sigmoid'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=None)
model.fit(X_train, Y_train, epochs=10, verbose=1)

# #perform auto-evaluation
# loss = model.evaluate(X_test, Y_test, verbose=1)
# print('Loss = ', loss)

#perform prediction (let's eye-ball the results)
predictions = model.predict(X_test)

acc = 0
for i in np.arange(len(predictions)):
    acc += np.math.pow(Y_test[i] - predictions[i], 2)
    print('Actual: ', Y_test[i], ', Predicted: ', predictions[i])

print('Computed loss: ', acc/len(Y_test))
