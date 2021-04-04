# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 07:25:47 2020

@author: DHRUV
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('column_2C.dat', sep='\s+',header=None)
df.columns = ['a','b','c','d','e','f','O']
print(df)

"""
Note : For the plots the labels AB and NO have been kept as AB and NO and have not
yet been converted to the binary 0/1 labels (0=NO,1=AB).

For plots we assume AB and NO are sufficient for visualization.

HW1 (b) (i)

Assumptions for scatterplots : The scatterplots are drawn with an independent variable
on the x-axis, an independent variable on the y-axis and the dependent variable 
(Output : AB/NO) deciding the hue. 

Independent Variables : a,b,c,d,e,f
Dependent Variable : O

As each graph has O, we have a total of 15 scatter plots, assuming that once we have 
the combination 
x-axis : a, y-axis : b
we don't need to consider
x-axis : b, y-axis : a

"""

sns.relplot(data=df, x='a', y='b', hue='O')
plt.show()
sns.relplot(data=df, x='a', y='c', hue='O')
plt.show()
sns.relplot(data=df, x='a', y='d', hue='O')
plt.show()
sns.relplot(data=df, x='a', y='e', hue='O')
plt.show()
sns.relplot(data=df, x='a', y='f', hue='O')
plt.show()
sns.relplot(data=df, x='b', y='c', hue='O')
plt.show()
sns.relplot(data=df, x='b', y='d', hue='O')
plt.show()
sns.relplot(data=df, x='b', y='e', hue='O')
plt.show()
sns.relplot(data=df, x='b', y='f', hue='O')
plt.show()
sns.relplot(data=df, x='c', y='d', hue='O')
plt.show()
sns.relplot(data=df, x='c', y='e', hue='O')
plt.show()
sns.relplot(data=df, x='c', y='f', hue='O')
plt.show()
sns.relplot(data=df, x='d', y='e', hue='O')
plt.show()
sns.relplot(data=df, x='d', y='f', hue='O')
plt.show()
sns.relplot(data=df, x='e', y='f', hue='O')
plt.show()

"""
HW1 (b) (ii)

Assumptions for the box plots : The box plots have been drawn with an independent 
variable on the y-axis (numeric) and the output/dependent variable (O:AB/NO) on the
x-axis which is again, obviously categoric.

So we have a total of 6 box plots of an independent variable (a,b,c,d,e,f) against
a dependent variable (O)

"""

sns.boxplot(data=df,x='O',y='a')
plt.show()
sns.boxplot(data=df,x='O',y='b')
plt.show()
sns.boxplot(data=df,x='O',y='c')
plt.show()
sns.boxplot(data=df,x='O',y='d')
plt.show()
sns.boxplot(data=df,x='O',y='e')
plt.show()
sns.boxplot(data=df,x='O',y='f')
plt.show()


"""

Continued


"""




