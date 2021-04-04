# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:07:38 2020

@author: DHRUV
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
HW1 b (iii)

"""

df=pd.read_csv('column_2C.dat', sep='\s+',header=None)
df.columns = ['a','b','c','d','e','f','O']
print(df)
df.loc[(df.O == 'AB'),'O']=1            
df.loc[(df.O == 'NO'),'O']=0            
print(df)
column_names = ['a','b','c','d','e','f','O']
i_dtest = np.arange(100)
i_dtraining = np.arange(210)
df_test = pd.DataFrame(columns=column_names, index=i_dtest)
df_training = pd.DataFrame(columns=column_names,index=i_dtraining)
print(df_test)
print(df_training)
r=np.arange(140)
s=np.arange(start=140,stop=210)

for i in r:
    df_training.loc[i]=df.loc[i]

    
for j in s:
    df_training.loc[j]=df.loc[j+70]

print(df_training)

u=np.arange(70)
v=np.arange(start=70,stop=100)

for k in u:
    df_test.loc[k]=df.loc[k+140]

for l in v:
    df_test.loc[l]=df.loc[l+210]

print(df_test)

df.to_csv('Data.csv')
df_training.to_csv('TrainingData.csv')
df_test.to_csv('TestData.csv')


"""
Code Explanation

We've read the data file first, converted the column headers into (a,b,c,d,e,f,O) and 
printed it.

Then we've converted the AB entry in O (Output) to 1 (Class 1, Numeric)
And the NO in O into 0 (Class 0, Numeric)

Then we've defined empty dataframes df_test and df_training to copy appropriate rows from
original dataframe df

Their column headers are same as df and row indices are:
    
df_training : 0 to 209 (210 rows)
df_test : 0 to 99 (100 rows)

TO COPY rows of df to df_training: To accomplish this to numpy arrays r and s are defined.
Over two FOR loops, r copies the first 140 rows of df to df_training and then for its last 
70 rows (from 140 to 209), s copies df rows from 210 to 279 index (corresponding to first
70 class 0 entries) into df_training. This completes formation of Training Data.

TO COPY rows of df to df_test: Similar method using u and v and two FOR loops. We copy 
remaining class 1 and class 0 data into df_test. This completes formation of Test Data.

Finally we export df, df_training and df_test to Data, TrainingData, TestData .csv files
for use in subsequent parts of HW1

"""
