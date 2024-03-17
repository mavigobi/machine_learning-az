# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:55:24 2023

@author: mavigobi
"""

# CATEGORICAL DATA TEMPLATE
# Before to define X_train,X_tets,Y_train & y_test.

# Import dataset
dataset = pd.read_csv('Data.csv');

# Import all files an all columns except last column
# Parameter ".values" give values of dataset
# Capital letters for matrix and lowercase letter for vectors
X = dataset.iloc[:, :-1].values; # Dependence
y = dataset.iloc[:, 3].values; # Independent

# Categorical data processing
# Two variables's type: categorical and ordinary
# This case, categorical variables

"""# It isn't compatible version
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:,0]=labelencoder_X.fit_transform(X[:,0])"""

# This version
from sklearn import preprocessing
le_X = preprocessing.LabelEncoder()
X[:, 0] = le_X.fit_transform(X[:, 0])

# Dummy variables: COUNTRY

"""# It isn't compatible version
#onehotenconder = OneHotEncoder(categories = 'auto', drop= 'first')
#X = onehotenconder.fit_transform(X).toarray()"""

# Dummy variables: COUNTRY (this version)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
 
ct_X = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'
    )
X = np.array(ct_X.fit_transform(X), dtype=np.float_)

# Ordinal variables: Purchase
le_y = preprocessing.LabelEncoder()
y = le_y.fit_transform(y)