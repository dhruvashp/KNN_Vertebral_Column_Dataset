# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 21:19:29 2020

@author: DHRUV
"""

"""
HW1 1(d)
i. A.

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
knn = KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='auto',metric='minkowski',p=1)
knn.fit(X,y)
Z=df_test.drop(columns=['O'])
test_prediction = knn.predict(Z)
print(test_prediction)
U=df_train.drop(columns=['O'])
train_prediction = knn.predict(U)
print(train_prediction)


"""
This code performs KNN (Minkowski Distance Metric) as p=1 (Manhattan hence)

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

All this is only for k=5 for Minkowski p=1, Manhattan

"""

k = np.arange(1,197,5)
print(k)

size_k = np.size(k)
print(size_k)

ite=np.arange(size_k)
print(ite)

Test_Error_Vector = np.empty([size_k])
Train_Error_Vector = np.empty([size_k])

for h in ite:
    knn = KNeighborsClassifier(n_neighbors=k[h],weights='uniform',algorithm='auto',metric='minkowski',p=1)
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
For Manhattan (Minkowski with p=1)

k=6 or k=16 seem like optimal points

k=6 has
Test Error = 0.11
Training Error = 0.138

k=16 has
Test Error = 0.12
Training Error = 0.1333

Either one of the two is sufficiently optimal.
We can choose k*=6 as our optimal point

so k*=6

"""


























    

