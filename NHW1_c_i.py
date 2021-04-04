# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:02:32 2020

@author: DHRUV
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as lrn
from sklearn.neighbors import KNeighborsClassifier

df_train=pd.read_csv('TrainingData.csv',index_col=0)
print(df_train)

df_test=pd.read_csv('TestData.csv',index_col=0)
print(df_test)


"""
HW1 1 (c) (i)

KNN

We'll use in-built functions 

"""
X=df_train.drop(columns=['O'])
y=df_train['O']
knn = KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='auto',metric='minkowski',p=2)
knn.fit(X,y)
l=np.array([[33.84,5.07,36.64,28.77,123.95,-0.20]])
mock = pd.DataFrame(l)
print(mock)
prediction = knn.predict(mock)
print(prediction)

"""
p=2 makes the distance metric Euclidean

All other functions used which are available in-built

"""
Z=df_test.drop(columns=['O'])
test_prediction = knn.predict(Z)
print(test_prediction)

"""
The above code gives the prediction on the test data using the KNN algorithm using
in built functions. Using various other attributes of the inbuilt KNeighborsClassifier
function we can perform different sorts of analysis on the features and predictions of/on our
dataset (test and training both).


"""




