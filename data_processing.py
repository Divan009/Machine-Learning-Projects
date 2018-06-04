# -*- coding: utf-8 -*-
"""
Created on Thu May 31 03:54:24 2018

@author: lenovo
"""

#!/usr/bin/env python
#Data Processing

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing Datasets
dataset = pd.read_csv('Data.csv')

#creating Matrix
X = dataset.iloc[:,:-1].values
#creating dependent var vector
Y = dataset.iloc[:,3].values   #choosing last col.

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis = 0)
#fit imputer object to our data X(matrix of feat. X)
# we want to add mean data only where data is missing, therefore, we choose 
#the whole col.(choose only 1 & 2 col.)
imputer = imputer.fit(X[:,1:3])  #to make our imputer obj fit into the data X
#we need to replace missing data of Matrix X by the mean of the resp. col.
X[:, 1:3] = imputer.transform(X[:, 1:3]) 

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder #dummy_var
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) #returns country col. encoded
#creating dummy var for countries
onehotencoder = OneHotEncoder(categorical_features=[0])#specifying col[0]
X = onehotencoder.fit_transform(X).toarray()
#creating dummy var for purchases(dependent col)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#splitting datset into training & testing set
from sklearn.cross_validation import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)













