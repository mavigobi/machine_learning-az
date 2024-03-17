# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:22:41 2023

@author: mavigobi
"""

# POLYNOMIAL REGRESSION

# PRE PROCESSING DATA

# How to import  libraries 
import numpy as np;# Mathematical tool
import matplotlib.pyplot as plt; # Graphics tool
import pandas as pd; # Read data

# Import dataset
dataset = pd.read_csv('Position_Salaries.csv');
X = dataset.iloc[:, 1:2]
y = dataset.iloc[:,2]

# IN THIS CASE, THE VARIABLES SHOULDN'T BE DIVIDED

# Overfitting problem / Split data
# Dataset: training set & test set
"""
from sklearn.model_selection import train_test_split # four variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # It is common to sacrifice 20% of the data for the test
"""

# Variable scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)""" 

# LINEAR REGRESSION DATA
from sklearn.linear_model import LinearRegression
l_regression = LinearRegression()
l_regression.fit(X, y)

# POLYNOMIAL REGRESSION DATA
from sklearn.preprocessing import PolynomialFeatures
poly_regression = PolynomialFeatures(degree=4) # polynomial regression ax^2 + bx + c
X_poly = poly_regression.fit_transform(X) 
lin_regression = LinearRegression() # linear regression of polynomical data regression
lin_regression.fit(X_poly,y)

# DISPLAY OF LINEAR REGRESSION MODEL
plt.scatter(X, y, color = "blue")
plt.plot(X, l_regression.predict(X), color = "red")
plt.title("Modelo de Regresión Lineal")
plt.xlabel("Sueldo anual del empleado")
plt.ylabel("Nivel laboral del empleado")
plt.show()

# DISPLAY OF POLYNOMIAL REGRESSION MODEL
X_grid = np.arange(min(X["Level"]), max(X["Level"]), 0.01, dtype= float) 
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = "blue")
plt.plot(X_grid, lin_regression.predict(poly_regression.fit_transform(X_grid)), color = "red")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Sueldo anual del empleado")
plt.ylabel("Nivel laboral del empleado")
plt.show()

# PREDICTION OF LINEAR REGRESSION AND POLYNOMIAL REGRESSION
l_regression.predict([[6.5]])
lin_regression.predict(poly_regression.fit_transform([[6.5]]))

