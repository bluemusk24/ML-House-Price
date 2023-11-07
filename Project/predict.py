
# # Load the file

import pickle

import numpy as np
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'LR.bin'

with open(model_file, 'rb') as f_in:
    dv, LR = pickle.load(f_in)

app = Flask('price')

@app.route('/predict', methods=['POST'])
def predict():

    house_ten = request.get_json()

    X_test = dv.transform(house_ten)
    y_pred = LR.predict(X_test)

    predicted_price = np.expm1(y_pred)
    result = {'average price of house_ten is' : (float(predicted_price.round(3)))}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8080)









#dv, LR = train(df_full_train, y_full_train)
#rmse = predict(df_test, dv, LR)
#rmse


#house_ten = df_test.iloc[10].to_dict()

#df_house_ten = pd.DataFrame([house_ten])

#dict_ten = df_house_ten.to_dict(orient='records')

#X_ten = dv.transform(dict_ten)

#y_pred = LR.predict(X_ten)

#predicted_price = np.expm1(y_pred[0])

#actual_price = np.expm1(y_test[10])

#print('predicted_price :', predicted_price.round(3))
#print('actual_price :', actual_price.round(3))