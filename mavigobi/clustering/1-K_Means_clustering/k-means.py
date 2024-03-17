# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:30:16 2024

@author: mavigobi
"""

# K-MEANS CLUSTERING

#IMPORT NECESSARY LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# LOAD DATASET
dataset = pd.read_csv("Mall_Customers.csv");
X = dataset.iloc[:,3:5].values
y = dataset.iloc[:,4].values

# METHOD OF ELBOW GRAPH REPRESENTATION FOR K-MEANS CLUSTERING

# The goal is to find the optimal point k
from sklearn.cluster import KMeans
wcss = [] # empty list
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

    
# VISUALITATION OF ELBOW GRAPH
plt.plot(range(1,11), wcss)
plt.title("Método del codo para conocer k")
plt.xlabel("Número de Clusters")
plt.ylabel("WCSS")
plt.show()

# METHOD OF K-MEANS 
kmeans = KMeans(n_clusters=5, init="k-means++", n_init= 10, max_iter=300, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# VISUALITATION OF DATA SET 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1], s= 100, c = "red", label ="Cluster 1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1], s= 100, c = "blue", label ="Cluster 2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1], s= 100, c = "orange", label ="Cluster 3")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3,1], s= 100, c = "green", label ="Cluster 4")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4,1], s= 100, c = "gray", label ="Cluster 5")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 150, c = "purple", label="Baricentos")
plt.title("Cluster de clientes")
plt.legend()
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()