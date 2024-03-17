# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:18:36 2023

@author: mavigobi
"""

# GENERAL REGRESSION TEMPLATE

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

# GENERAL REGRESSION MODEL



# PREDICTION OF OWN REGRESSION MODEL
y_pred = regression.predict([[6.5]])

# VISUALITATION OF POLYNOMIAL REGRESSION MODEL
X_grid = np.arange(min(X), max(X), 0.01, dtype= float) 
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = "blue")
plt.plot(X_grid, regression.predict(X_grid), color = "red")
plt.title("Modelo de Regresión")
plt.xlabel("Sueldo anual del empleado")
plt.ylabel("Nivel laboral del empleado")
plt.show()




