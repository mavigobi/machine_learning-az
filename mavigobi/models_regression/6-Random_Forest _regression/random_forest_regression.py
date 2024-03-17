# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:44:22 2023

@author: mavigobi
"""

# RANDOM TREE REGRESSION 

# DATA PRE PROCESSING TEMPLATE

# How to import  libraries 
import numpy as np;# Mathematical tool
import matplotlib.pyplot as plt; # Graphics tool
import pandas as pd; # Read data

# Import dataset
dataset = pd.read_csv('Position_Salaries.csv');
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

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

# RANDOM FOREST REGRESSION MODEL
from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(n_estimators = 300, random_state = 0) 
# n_estimator variable makes the forest grow but the number of steps does not increase
# the estimated value for n_estimators = 10 is worse than n_estimators = 100
# n_estimators = 300 is the best prediction
regression.fit(X,y)

# PREDICTION OF OWN REGRESSION MODEL
y_pred = regression.predict([[6.5]])

# VISUALITATION OF RANDOM FOREST REGRESSION MODEL
X_grid = np.arange(min(X), max(X), 0.01, dtype= float) 
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = "blue")
plt.plot(X_grid, regression.predict(X_grid), color = "red")
plt.title("Random Forest (Modelo de rRegresión)")
plt.xlabel("Sueldo anual del empleado")
plt.ylabel("Nivel laboral del empleado")
plt.show()




