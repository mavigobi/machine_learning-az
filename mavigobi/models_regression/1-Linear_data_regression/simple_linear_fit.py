# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:02:33 2023

@author: mavigobi
"""

# REGRESION LINEAR SIMPLE

# PRE PROCESSING DATA

# How to import  libraries 
import numpy as np;# Mathematical tool
import matplotlib.pyplot as plt; # Graphics tool
import pandas as pd; # Read data

# Import dataset
dataset = pd.read_csv('Salary_Data.csv');

# Import all files an all columns except last column
# Parameter ".values" give values of dataset
# Capital letters for matrix and lowercase letter for vectors
X = dataset.iloc[:,:-1].values; # Dependence
y = dataset.iloc[:,1].values; # Independent

# Overfitting problem / Split data
# Dataset: training set & test set
from sklearn.model_selection import train_test_split # four variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) # It is common to sacrifice 20% of the data for the test

# LINEAR REGRESSION

# Linear regression not need to scale variables.
# Variable scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)""" 

# Linear regression: train variables
from sklearn.linear_model import LinearRegression
regression = LinearRegression() #least squares technique
regression.fit(X_train, y_train, sample_weight=None)

# Linear regression: test variables
y_pred = regression.predict(X_test)

# DISPLAY OF DATA REGRESSION

# Data display: train variables
plt.scatter(X_train, y_train, color = "blue")
plt.plot(X_train, regression.predict(X_train), color = "red")
plt.title("Salary vs Experience (Train Data)")
plt.xlabel("Exployee experience  (years).")
plt.ylabel("Employee salary ($)")
plt.show()

# Data display: test and train data regression are identical
plt.scatter(X_test, y_test, color = "blue")
plt.plot(X_train, regression.predict(X_train), color = "red")
plt.title("Salary vs Experience (Train Data)")
plt.xlabel("Exployee experience  (years).")
plt.ylabel("Employee salary ($)")
plt.show()

