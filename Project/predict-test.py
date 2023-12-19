#!/usr/bin/env python
# coding: utf-8


import requests

#host = 'house-price-env.eba-rkqvdcb6.us-east-1.elasticbeanstalk.com'           #domain name from awsebs

#url = 'http://{host}/predict'

#url = 'http://localhost:8080/predict'

url = 'http://a26f9963717184b76b0255b90cb09805-1892142795.us-east-1.elb.amazonaws.com/predict'


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



house_nine = {'avg._area_income': 71408.29574076342,
 'avg._area_house_age': 6.565006381455809,
 'avg._area_number_of_rooms': 7.232609527481563,
 'avg._area_number_of_bedrooms': 6.21,
 'area_population': 49463.04926333042,
 'address': '8368 jacqueline run apt. 172\ncollinsfort, sc 29889-1351'

}


requests.post(url, json=house_nine)

response = requests.post(url, json=house_nine).json()
                                            
print(response)