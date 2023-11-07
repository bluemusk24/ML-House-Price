#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error

# Parameters

output_file = 'LR.bin'

# Data preparation

df = pd.read_csv("C:\\Users\\emman\\kaggle datasets\\USA_Housing.csv")

df.columns = df.columns.str.lower().str.replace(' ','_')

df['price'] = np.log1p(df['price'])

object = list(df.dtypes[df.dtypes == 'object'].index)

for col in object:
    df[col] = df[col].str.lower()
    
df.index.names = ['index_col']

dv = DictVectorizer(sparse=False)

LR = LinearRegression()

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

dict_full_train = df_full_train.to_dict(orient='records')

X_full_train = dv.fit_transform(dict_full_train)

y_full_train = df_full_train['price'].values

y_test = df_test['price'].values

del df_full_train['price']

del df_test['price']

LR.fit(X_full_train,y_full_train)

# Training function for the above

def train(df_full_train, y_full_train):
    dict_full_train = df_full_train.to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(dict_full_train)
    
    LR = LinearRegression()
    LR.fit(X_full_train,y_full_train)
    
    return dv, LR

# RMSE predict function

def predict(df, dv, LR):
    dict_test = df_test.to_dict(orient='records')
    
    X_test = dv.transform(dict_test)
    y_pred = LR.predict(X_test)
    
    mse = mean_squared_error(y_pred, y_test)
    rmse = np.sqrt(mse)
    
    return rmse

dv, LR = train(df_full_train, y_full_train)
rmse = predict(df_test, dv, LR)

print(rmse)


# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, LR), f_out)
