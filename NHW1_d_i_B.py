# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 21:19:29 2020

@author: DHRUV
"""

"""
HW1 1(d)
i. B.

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
This code performs KNN (Manhattan Distance Metric) as p=1

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

All this is only for k=5 for Manhattan

"""

k = np.arange(1,197,5)
print(k)

size_k = np.size(k)
print(size_k)

ite=np.arange(size_k)
print(ite)
power_p=[10**0.1,10**0.2,10**0.3,10**0.4,10**0.5,10**0.6,10**0.7,10**0.8,10**0.9,10**1]
p_range=np.size(power_p)
p_index=np.arange(p_range)

Test_Error_Vector = np.empty([size_k,p_range])
Train_Error_Vector = np.empty([size_k,p_range])

for h in ite:
    for iota in p_index:
      knn = KNeighborsClassifier(n_neighbors=k[h],weights='uniform',algorithm='auto',metric='minkowski',p=power_p[iota])
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
      Test_Error_Vector[h,iota]=(p/100)
      Train_Error_Vector[h,iota]=(q/210)
        

print(Test_Error_Vector)
print(Train_Error_Vector)

"""

The Error Vectors are no longer vectors but are matrices. The row corresponds to a value of k and 
column corresponds to a value of p

"""

Error_Test=pd.DataFrame(data=Test_Error_Vector,index=k,columns=['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10'])
print(Error_Test)
Error_Train=pd.DataFrame(data=Train_Error_Vector,index=k,columns=['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10'])
print(Error_Train)

"""
Converting array to a more readable dataframe of errors, p's written sequential

"""

Error_Test.to_csv('Error_Test.csv')
Error_Train.to_csv('Error_Train.csv')

"""
Converting to csv to visualize and decide p

"""


"""
Upon examining the data, there doesn't seem to be an "overall" best value of p

Now to examine trend of p corresponding to k*=16 selected in HW1_d_i_A

We'll draw error vs p curves for a given k*=16
"""

e_test = Error_Test.loc[16]
e_train = Error_Train.loc[16]



y=[1,2,3,4,5,6,7,8,9,10]

plt.plot(y,e_test,'r')
plt.plot(y,e_train,'b')
plt.xlabel('logp*10')
plt.ylabel('Errors, red = test, blue = training')


"""
From the plot for k*=16,
we can clearly see that 
logp = 0.6 minimizes the training error and the test error though not minimized, is
sufficiently low
logp= 0.8 and logp=1 also minimize the test error, but the training error does increase moderately

So
logp=0.6 is an optimal point 

(we can also include 0.8 and 1 as they don't perform badly either)

"""















    

