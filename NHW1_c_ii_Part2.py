# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:27:24 2020

@author: DHRUV
"""

"""
HW1 c(ii) Part 1

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as lrn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

df_train=pd.read_csv('TrainingData.csv',index_col=0)
print(df_train)

df_test=pd.read_csv('TestData.csv',index_col=0)
print(df_test)
X=df_train.drop(columns=['O'])
y=df_train['O']
knn = KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='auto',metric='minkowski',p=2)
knn.fit(X,y)
Z=df_test.drop(columns=['O'])
test_prediction = knn.predict(Z)
print(test_prediction)
U=df_train.drop(columns=['O'])
train_prediction = knn.predict(U)
print(train_prediction)


"""
This code performs KNN (Euclidean Distance Metric) and predicts the output over the Test
Data and the Training Data, both. Obviously the classification is done by majority polling between frequency of occurence
of 1's and 0's in the neighborhood of a point. Here note that k=5.

"""

test=np.arange(100)
train=np.arange(210)

p=0
q=0


test_y = df_test['O'].to_numpy()
train_y=df_train['O'].to_numpy()

print(test_y)
print(train_y)

for i in test:
    if test_y[i] != test_prediction[i]:
        p=p+1

for f in train:
    if train_y[f] != train_prediction[f]:
        q=q+1

print(p)
print(q)

test_error=p/100
train_error=q/210

print(test_error)
print(train_error)

"""
p are the test prediction mismatches
q are the training prediction mismatches

test_error and train_error are, respectively, the test error and the training error

All this is only for k=5

"""

k = np.arange(208,0,-3)
print(k)

size_k = np.size(k)
print(size_k)

ite=np.arange(size_k)
print(ite)

Test_Error_Vector = np.empty([size_k])
Train_Error_Vector = np.empty([size_k])

for h in ite:
    knn = KNeighborsClassifier(n_neighbors=k[h],weights='uniform',algorithm='auto',metric='minkowski',p=2)
    knn.fit(X,y)
    test_prediction = knn.predict(Z)
    train_prediction = knn.predict(U)
    t=0
    q=0
    p=0
    q=0 
    for t in test:
        if test_y[t] != test_prediction[t]:
            p=p+1
    for r in train:
        if train_y[r] != train_prediction[r]:
            q=q+1
    Test_Error_Vector[h]=(p/100)
    Train_Error_Vector[h]=(q/210)
        

print(Test_Error_Vector)
print(Train_Error_Vector)

plt.plot(k,Test_Error_Vector,'r')
plt.plot(k,Train_Error_Vector,'b')
plt.xlabel("K Values")
plt.ylabel("Errors (Blue - Training Error, Red - Test Error)")
plt.show()

"""
From the curves and also from the error vectors we can see the following:
Roughly K values in the range of 4 to 40 (approximate) seem the best for 
minimizing both the Training and Test Errors.

We'll take the optimal value to be k*=4, as test error and training error are both quite 
low (test error 0.06, training error 0.1428)

However k=10 is also an equally valid choice for k* as test error (0.1) and training error
(0.1333) are also quite low.

We'll go with k*=4 

(optimal k could be 4 and also 10, k should lie between 4-40, k*=4 selected)

"""























    