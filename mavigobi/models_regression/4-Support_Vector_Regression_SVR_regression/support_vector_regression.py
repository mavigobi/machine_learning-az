# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 20:47:59 2023

@author: mavigobi
"""

# SUPPORT VECTOR REGRESSION (SVR)


# DATA PRE PROCESSING 

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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1)) 

# GENERAL REGRESSION MODEL
from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(X,y)


# PREDICTION OF OWN REGRESSION MODEL
y_pred = sc_y.inverse_transform([regression.predict(sc_X.transform(np.array([[6.5]])))])
"""y_pred = regression.predict(sc_X.transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred)"""

# VISUALITATION OF POLYNOMIAL REGRESSION MODEL
#X_grid = np.arange(min(X), max(X), 0.01, dtype= float) 
#X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = "blue")
plt.plot(X, regression.predict(X), color = "red")
plt.title("Modelo de Regresión")
plt.xlabel("Sueldo anual del empleado")
plt.ylabel("Nivel laboral del empleado")
plt.show()




