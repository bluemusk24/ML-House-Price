# ML-House-Price

ML-House-price-project description:

1.  This project was on predicting the prices of house based on following features: Avg. Area Income, Avg. Area House Age, Avg. Area Number of Rooms, Avg. Area Number of Bedrooms, Area Population, Price, Address.

2.  The instructions on how to execute this project is stated as follows: (a) I downloaded the USA_ Housing dataset from Kaggle (b) I read the dataset on jupyter notebook, cleaned the data, did exploratory data analysis and feature engineering, built and trained models- linear regression and ridge regression for my training and validation datasets to predict the prices of houses. Finally, I chose the model with the lowest root mean squared error for my testimg dataset.

3.  I downloaded my ipynb notebook as a python script on visual studio code, named it train.py, did some editing and created a webservice with flask (predict.py) to predict house prices on an unknown dataset. I ensured the model predicted right.

4.  Because I use windows operating system, I downloaded windows subsystem for linux (WSL) to ensure Linux command work on windows computer. Example of such commands are: gunicorn --bind localhost:8080 predict:app

5.  I created a virtual environment for my model to include all its dependencies with pip install pipenv, pipenv install numpy, scikit-learn==1.0.2 flask inorder to download the Pipfile and Pipfile.lock which contained all dependencies.

6.  From docker python image, I got a python image with tag- python:3.10.12-slim, use this code:docker run -it --rm python:3.10.12-slim to download it while I ensured my docker desktop was up and running, created a docker file to overwrite the downloaded python image-docker build -t project-test ., docker run -it --rm --entrypoint=bash project-test, and finally run it with: docker run -it --rm -p 8080:8080 project-test.

7.  Finally, I deployed my docker container on AWS elasticbeanstalk using command line interface: pipenv install awsebcli --dev, pipenv shell, eb init --help, eb init -p docker -r us-east-1 house-price, ls -a, less .elasticbeanstalk/config.yml, eb local run --port 8080, eb create house-price-env, and terminated using eb terminate churn-serving-env.
