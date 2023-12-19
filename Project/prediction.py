import pickle
import numpy as np


model_file = 'LR.bin'

with open(model_file, 'rb') as f_in:
    dv, LR = pickle.load(f_in)

house_500 = {'avg._area_income': 71408.29574076342,
 'avg._area_house_age': 6.565006381455809,
 'avg._area_number_of_rooms': 7.232609527481563,
 'avg._area_number_of_bedrooms': 6.21,
 'area_population': 49463.04926333042,
 'address': '8368 jacqueline run apt. 172\ncollinsfort, sc 29889-1351'
 }

X = dv.transform(house_500)
y_pred = LR.predict(X)

predicted_price = np.expm1(y_pred)

print('average price of house_500 is :', predicted_price.round(3))


