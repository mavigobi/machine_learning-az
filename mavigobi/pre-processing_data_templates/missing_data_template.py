# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:55:51 2023

@author: mavigobi
"""

# MISSIND DATA (Nan) TEMPLATE

# Import dataset
dataset = pd.read_csv('Data.csv');

# Import all files an all columns except last column
# Parameter ".values" give values of dataset
# Capital letters for matrix and lowercase letter for vectors
X = dataset.iloc[:, :-1].values; # Dependence
y = dataset.iloc[:, 3].values; # Independent

# Nas data processing
# from sklearn.preprocessing import Imputer not valid in recent version
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean');
imputer = imputer.fit(X[:, 1:3]);
X[:, 1:3] = imputer.transform(X[:, 1:3]);