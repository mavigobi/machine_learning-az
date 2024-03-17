# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:37:35 2023

@author: mavigobi
"""


# ORIGINAL BACKWARD ELIMINATION OF P-VALUE TEMPLATE 


# DATA PRE PROCESSING 
# How to import  libraries 
import numpy as np;# Mathematical tool
import matplotlib.pyplot as plt; # Graphics tool
import pandas as pd; # Read data

# Import dataset
dataset = pd.read_csv('50_Startups.csv');

# Import all files an all columns except last column
# Parameter ".values" give values of dataset
# Capital letters for matrix and lowercase letter for vectors
X = dataset.iloc[:, :-1].values; # Dependence
y = dataset.iloc[:, 4].values; # Independent

# Categorical data processing
# Two variables's type: categorical and ordinary
# This case, categorical variables

# This version
from sklearn import preprocessing
le_X = preprocessing.LabelEncoder()
X[:, 3] = le_X.fit_transform(X[:, 3])

# Dummy variables: COUNTRY

"""# It isn't compatible version
#onehotenconder = OneHotEncoder(categories = 'auto', drop= 'first')
#X = onehotenconder.fit_transform(X).toarray()"""

# Dummy variables: COUNTRY (this version)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
 
ct_X = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   
    remainder='passthrough'
    )
X = np.array(ct_X.fit_transform(X), dtype=np.float_)

# Avoid dummy variables (N-1)
X = X[:,1:]

# Overfitting problem / Split data
# Dataset: training set & test set
from sklearn.model_selection import train_test_split # four variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # It is common to sacrifice 20% of the data for the test

# Variable scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)""" 


# MULTIPLE LINEAR REGESSION
# Multiple Linear Regression of testing variables
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Optimiation of model
# Multiple linear regression: test variables
y_pred = regression.predict(X_test)


# BACKWARD ELIMINATION OF P-VALUE > 0.05 (personal method)
# The goal is to increase the accuracy of the data model
import statsmodels.formula.api as sm
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
