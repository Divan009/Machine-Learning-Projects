# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 16:43:26 2018

@author: lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('train.csv')
y = dataset.SalePrice
iowa_predictors = ['LotArea','YearBuilt','1stFlrSF', '2ndFlrSF','BedroomAbvGr','TotRmsAbvGrd','FullBath']
X = dataset[iowa_predictors]

#splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# =============================================================================
# from sklearn.tree import DecisionTreeRegressor
# #define model
# iowa_model = DecisionTreeRegressor()
# #fit model
# iowa_model.fit(X_train,y_train)
# 
# print("Making predictions for 5 houses:")
# print(X.head())
# print("The prices for each house are")
# print(iowa_model.predict(X.head()))
# 
# =============================================================================
#MAE predicting averaage error
from sklearn.metrics import mean_absolute_error

# =============================================================================
# def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
#     model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
#     model.fit(predictors_train, targ_train)
#     preds_val = model.predict(predictors_val)
#     mae = mean_absolute_error(targ_val,preds_val)
#     return(mae)
# 
# for max_leafs_nodes in [5,50,500,5000]:
#     my_mae = get_mae(max_leafs_nodes, X_train, X_test, y_train, y_test)
#     print("Max leaf nodes:%d \t\t MAE:%d" %(max_leafs_nodes, my_mae))
# =============================================================================
#the output we got is called "In-Sample" score

from sklearn.ensemble import RandomForestRegressor    
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
iowa_preds = forest_model.predict(X_test)
print(mean_absolute_error(y_test, iowa_preds))   
    
    





