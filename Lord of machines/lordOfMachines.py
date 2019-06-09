# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 05:53:10 2019

@author: lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


train = pd.read_csv('train.csv')
test = pd.read_csv("test.csv")
campaign = pd.read_csv('campaign_data.csv')

train.info()


# This way we have randomness and are able to reproduce the behaviour within this cell.
np.random.seed(13)

def impact_coding(data, feature, target='y'):
    '''
    In this implementation we get the values and the dictionary as two different steps.
    This is just because initially we were ignoring the dictionary as a result variable.
    
    In this implementation the KFolds use shuffling. If you want reproducibility the cv 
    could be moved to a parameter.
    '''
    n_folds = 10
    n_inner_folds = 5
    impact_coded = pd.Series()
    
    oof_default_mean = data[target].mean() # Gobal mean to use by default (you could further tune this)
    kf = KFold(n_splits=n_folds, shuffle=True)
    oof_mean_cv = pd.DataFrame()
    split = 0
    for infold, oof in kf.split(data[feature]):
            impact_coded_cv = pd.Series()
            kf_inner = KFold(n_splits=n_inner_folds, shuffle=True)
            inner_split = 0
            inner_oof_mean_cv = pd.DataFrame()
            oof_default_inner_mean = data.iloc[infold][target].mean()
            for infold_inner, oof_inner in kf_inner.split(data.iloc[infold]):
                # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)
                oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()
                impact_coded_cv = impact_coded_cv.append(data.iloc[infold].apply(
                            lambda x: oof_mean[x[feature]]
                                      if x[feature] in oof_mean.index
                                      else oof_default_inner_mean
                            , axis=1))

                # Also populate mapping (this has all group -> mean for all inner CV folds)
                inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
                inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
                inner_split += 1

            # Also populate mapping
            oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
            oof_mean_cv.fillna(value=oof_default_mean, inplace=True)
            split += 1
            
            impact_coded = impact_coded.append(data.iloc[oof].apply(
                            lambda x: inner_oof_mean_cv.loc[x[feature]].mean()
                                      if x[feature] in inner_oof_mean_cv.index
                                      else oof_default_mean
                            , axis=1))

    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean

### Prepare dataset for encoding
train = train.merge(campaign,on = 'campaign_id',how = 'left')
test = test.merge(campaign, on = 'campaign_id',how = 'left')
train['user_id'] = train['user_id'].apply(str)
train['campaign_id'] = train['campaign_id'].apply(str)
test['user_id'] = test['user_id'].apply(str)
test['campaign_id'] = test['campaign_id'].apply(str)

train = train.drop(['email_body', 'subject', 'email_url','send_date'],axis =1 )
test = test.drop(['email_body', 'subject', 'email_url','send_date'],axis =1 )

features = train.columns
numeric_features = []
categorical_features = []

for dtype, feature in zip(train.dtypes, train.columns):
    if dtype == object:
        #print(column)
        #print(train_data[column].describe())
        categorical_features.append(feature)
    else:
        numeric_features.append(feature)
categorical_features

# Apply the encoding to training and test data, and preserve the mapping
impact_coding_map = {}
for f in categorical_features[1:]:
    print("Impact coding for {}".format(f))
    train["impact_encoded_{}".format(f)], impact_coding_mapping, default_coding = impact_coding(train, f,'is_click')
    impact_coding_map[f] = (impact_coding_mapping, default_coding)
    mapping, default_mean = impact_coding_map[f]
    test["impact_encoded_{}".format(f)] = test.apply(lambda x: mapping[x[f]] if x[f] in mapping else default_mean, axis=1)

impact_coding_map = {}
for f in categorical_features[1:]:
    print("Impact coding for {}".format(f))
    train["impact_encoded_open_{}".format(f)], impact_coding_mapping, default_coding = impact_coding(train, f,'is_open')
    impact_coding_map[f] = (impact_coding_mapping, default_coding)
    mapping, default_mean = impact_coding_map[f]
    test["impact_encoded_open_{}".format(f)] = test.apply(lambda x: mapping[x[f]] if x[f] in mapping else default_mean , axis=1)


train.to_csv( 'impact_encoded_train.csv',index = False)
test.to_csv( 'impact_encoded_test.csv',index = False)
 

























