# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 19:52:20 2020

@author: DHRUV
"""
"""
HW1 c(iii)

"""
"""
Assumptions
We assume that despite the reduction in training data, each training data is absolute
i.e. once it is shortened the remanant data won't be added to the test data set.
 
"""
import math
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
test_compare = df_test['O'].to_numpy()
train_compare_interim = df_train['O'].to_numpy()
column_names = ['a','b','c','d','e','f','O']

N=np.arange(10,211,10)
print(N)

test_error = np.zeros((21,42))
training_error = np.zeros((21,42))



size=np.size(N)
print(size)
ind=np.arange(size)

for i in ind:
    a=math.floor(N[i]/3)
    b=N[i]-a
    df_training_modified = pd.DataFrame(columns=column_names,index=np.arange(N[i]))
    train_compare1=train_compare_interim[0:b]
    train_compare2=train_compare_interim[140:a+140]
    train_compare = np.concatenate((train_compare1,train_compare2),axis=None)
    l=np.arange(a)
    m=np.arange(b)
    j=0
    t=0
    for j in m:
        df_training_modified.loc[j]=df_train.loc[j]
    for t in l:
        df_training_modified.loc[t+b]=df_train.loc[140+t]
    
    X=df_training_modified.drop(columns=['O'])
    y=df_training_modified['O']
    y=y.astype('int')
    
    
    k = np.arange(1,N[i]+1,5)
    size_k = np.size(k)
    ind_k = np.arange(size_k)
    for q in ind_k:
        knn = KNeighborsClassifier(n_neighbors=k[q],weights='uniform',algorithm='auto',metric='minkowski',p=2)
        knn.fit(X,y)
        test_prediction = knn.predict(df_test.drop(columns=['O']))
        training_prediction = knn.predict(df_training_modified.drop(columns=['O']))
        testing=np.arange(100)
        training = np.arange(N[i])
        alpha = 0
        beta = 0
        test_interim = 0
        train_interim = 0
        for alpha in testing:
            if test_compare[alpha] != test_prediction[alpha]:
                test_interim = test_interim + 1
        
        for beta in training:
            if train_compare[beta] != training_prediction[beta]:
                train_interim = train_interim + 1
        
        test_error[i][q]=test_interim/100
        training_error[i][q]=train_interim/N[i]
    

print(test_error)
print(training_error)

"""
converting test_error and training_error into dataframes with more
relevant indexing

"""

mega_k = np.arange(1,211,5)

test_1 = pd.DataFrame(data=test_error, index=N,columns=mega_k)
train_1 = pd.DataFrame(data=training_error, index=N,columns=mega_k)

print(test_1)
print(train_1)

test_1.to_csv('Test_1.csv')            
train_1.to_csv('Train_1.csv')                

"""
The row index of both test_1 and train_1 corresponds to the N value.
The column index to the k value.
As k changes for each N, this matrix, which was defined from a zero matrix, is obviously sparse.
Thus 0 values don't, in the dataframe, mean that the actual errors were zero, they
simply imply that k for that N was not defined and thus a zero initial entry was not
filled up.

Obviously training error is 0 for k=1, irrespective of N. For each N, k is only 
obviously defined at the very most equal to that value of N, and so entries where we have 0
and k>N were zero because of "non-existence"

0 values for k<=n undoubtedly imply that the error, in fact, was obtained zero. Note
that as k,N are not regular matrix indices over which these relationships hold, we have
no special "shape" of the matrix

"""

"""
plotting test error for some k, say, k=6 against every single N

"""

mock_test_rate = test_1[6].to_numpy()
print(mock_test_rate)
plt.plot(N,mock_test_rate)
plt.xlabel('N')
plt.ylabel('Test Error Rate, k=6')

"""

Obviously N being higher gives smallest error for k=6
Similar plots can be made on the same graph for different k values to obtain the best
overall "k", i.e. k that is the closest to the x-axis and has smaller overall errors over
N's entire range.

Also for each N, an optimal value for the k=k* can be obtained for the same graph, where
all k's are plotted for their Test Errors against the entire N range.

Additionally from the exported .csv files, we can also obtain the various
relevant information for k and N selection. Those .csv files contain all the test and training
errors for all the k and N values for this dataset.

Similar procedures can be done in combination with Training Errors or for Training errors
individually also, using the test_1 and train_1 datasets.

"""
