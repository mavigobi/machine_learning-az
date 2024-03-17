# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:26:29 2023

@author: mavigobi
"""

# DATA PRE PROCESSING TEMPLATE

# How to import  libraries 
import numpy as np;# Mathematical tool
import matplotlib.pyplot as plt; # Graphics tool
import pandas as pd; # Read data

# Import dataset
dataset = pd.read_csv('Data.csv');

# Import all files an all columns except last column
# Parameter ".values" give values of dataset
# Capital letters for matrix and lowercase letter for vectors
X = dataset.iloc[:, :-1].values; # Dependence
y = dataset.iloc[:, 3].values; # Independent

# Overfitting problem / Split data
# Dataset: training set & test set
from sklearn.model_selection import train_test_split # four variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # It is common to sacrifice 20% of the data for the test

# Variable scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)""" 

# x.transform(X) create and apply scaling to X
# x.fit_transform(X) apply scaling to X
# x.fit(X) create scaling to X but not apply