#!/usr/bin/env python
# coding: utf-8


import requests

host = 'house-price-env.eba-rkqvdcb6.us-east-1.elasticbeanstalk.com'           #domain name from awsebs

url = 'http://{host}/predict'

#url = 'http://localhost:8080/predict'


house_ten ={'avg._area_income': 55340.608735324255,
 'avg._area_house_age': 5.231696666386671,
 'avg._area_number_of_rooms': 5.614293707083566,
 'avg._area_number_of_bedrooms': 4.3,
 'area_population': 34112.97062220211,
 'address': '489 john locks\nwest kylestad, il 55787-7291'
 }




requests.post(url, json=house_ten)

response = requests.post(url, json=house_ten).json()
                                            
print(response)



