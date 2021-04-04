# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:51:20 2020

@author: DHRUV
"""

"""
HW1 c(ii) Part 2

"""

"""

k = 16 (square of k*=4)

"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as lrn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

df_train=pd.read_csv('TrainingData.csv',index_col=0)
print(df_train)

df_test=pd.read_csv('TestData.csv',index_col=0)
print(df_test)
X=df_train.drop(columns=['O'])
y=df_train['O']
knn = KNeighborsClassifier(n_neighbors=16,weights='uniform',algorithm='auto',metric='minkowski',p=2)
knn.fit(X,y)
Z=df_test.drop(columns=['O'])
test_prediction = knn.predict(Z)
print(test_prediction)
U=df_train.drop(columns=['O'])
train_prediction = knn.predict(U)
print(train_prediction)


"""
Confusion Matrix

"""


Test_Confusion=confusion_matrix(df_test['O'].to_numpy(),test_prediction)
Train_Confusion=confusion_matrix(df_train['O'].to_numpy(),train_prediction)
print(Test_Confusion)
print(Train_Confusion)


"""
True Positive
True Negative

"""

print('True Positive for Test is : ',Test_Confusion[1][1])
print('True Negative for Test is : ',Test_Confusion[0][0])
print('True Positive for Train is : ',Train_Confusion[1][1])
print('True Negative for Train is : ',Train_Confusion[0][0])

"""
Precision
"""


Precision_Test=precision_score(df_test['O'].to_numpy(),test_prediction)
Precision_Train=precision_score(df_train['O'].to_numpy(),train_prediction)

print('Precision for Test is : ',Precision_Test)
print('Precision for Train is : ',Precision_Train )




"""
F1 Score
"""

F1_Test=f1_score(df_test['O'].to_numpy(),test_prediction)
F1_Train=f1_score(df_train['O'].to_numpy(),train_prediction)

print('F1 for Test is : ',F1_Test)
print('F1 for Train is : ',F1_Train )
