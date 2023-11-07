#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error


# In[82]:


df = pd.read_csv("C:\\Users\\emman\\kaggle datasets\\USA_Housing.csv")

df.columns = df.columns.str.lower().str.replace(' ','_')

df['price'] = np.log1p(df['price'])

object = list(df.dtypes[df.dtypes == 'object'].index)

for col in object:
    df[col] = df[col].str.lower()
    
df.index.names = ['index_col']


# In[83]:


dv = DictVectorizer(sparse=False)

LR = LinearRegression()


# In[84]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[85]:


dict_full_train = df_full_train.to_dict(orient='records')

X_full_train = dv.fit_transform(dict_full_train)

y_full_train = df_full_train['price'].values

y_test = df_test['price'].values

del df_full_train['price']

del df_test['price']

LR.fit(X_full_train,y_full_train)


# In[86]:


def train(df_full_train, y_full_train):
    dict_full_train = df_full_train.to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(dict_full_train)
    
    LR = LinearRegression()
    LR.fit(X_full_train,y_full_train)
    
    return dv, LR


# In[87]:


def predict(df, dv, LR):
    dict_test = df_test.to_dict(orient='records')
    
    X_test = dv.transform(dict_test)
    y_pred = LR.predict(X_test)
    
    mse = mean_squared_error(y_pred, y_test)
    rmse = np.sqrt(mse)
    
    return rmse


# In[88]:


dv, LR = train(df_full_train, y_full_train)
rmse = predict(df_test, dv, LR)
rmse


# In[89]:


house_ten = df_test.iloc[10].to_dict()

df_house_ten = pd.DataFrame([house_ten])

dict_ten = df_house_ten.to_dict(orient='records')

X_ten = dv.transform(dict_ten)

y_pred = LR.predict(X_ten)

predicted_price = np.expm1(y_pred[0])

actual_price = np.expm1(y_test[10])

print('predicted_price :', predicted_price.round(3))
print('actual_price :', actual_price.round(3))


# # Save the model

# In[92]:


import pickle


# In[93]:


output_file = 'LR.bin'
output_file


# In[94]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, LR), f_out)


# # Load the file

# In[1]:


import pickle


# In[2]:


model_file = 'LR.bin'


# In[3]:


with open(model_file, 'rb') as f_in:
    dv, LR = pickle.load(f_in)


# In[4]:


dv, LR


# In[ ]:




