# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:28:21 2023

@author: mavigobi
"""

# CLASSIFICATOR REGRESSION TEMPLATE

# DATA PRE PROCESSING 

# How to import  libraries 
import numpy as np;# Mathematical tool
import matplotlib.pyplot as plt; # Graphics tool
import pandas as pd; # Read data

# Import dataset
dataset = pd.read_csv('Social_Network_Ads.csv');
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:,-1].values

# IN THIS CASE, THE VARIABLES SHOULDN'T BE DIVIDED

# Overfitting problem / Split data
# Dataset: training set & test set
"""from sklearn.model_selection import train_test_split # four variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) # It is common to sacrifice 20% of the data for the test"""

# Variable scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# CLASSIFIER REGRESSION MODEL
# MAKE HERE THE CLASSIFIER MODEL


# PREDICT OF OWN CLASSIFIER REGRESSION MODEL
y_pred = classifier.predict(X_test)

# CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
# it's not capitalized because it's not  a library, confusion_matrix is a function
cm = confusion_matrix(y_test, y_pred)
# the diagonal numbers are correct predictions: users who will buy or not 
# the not diagonal numbers are incorrect predictions: 8+3 data users wrong

# VISUALITATION OF LOGISTIC REGRESSION MODEL: TRAINING DATA
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min() -1, stop = X_set[:,0].max()+1, step =0.01),
                    np.arange(start = X_set[:,1].min() -1, stop = X_set[:,1].max()+1, step =0.01))
plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1],
                c = ListedColormap(('red','green'))(i), label = j) # label classifies if it's o or 1.
plt.title("Modelo de regresión logístico (training data)")
plt.xlabel("Edad")                
plt.ylabel("Sueldo estimado")
plt.legend()
plt.show()


# VISUALITATION OF LOGISTIC REGRESSION MODEL: TESTING DATA 
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min() -1, stop = X_set[:,0].max()+1, step =0.01),
                    np.arange(start = X_set[:,1].min() -1, stop = X_set[:,1].max()+1, step =0.01))
plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1],
                c = ListedColormap(('red','green'))(i), label = j)
plt.title("Modelo de regresión logístico (testing data)")
plt.xlabel("Edad")                
plt.ylabel("Sueldo estimado")
plt.legend()
plt.show()

