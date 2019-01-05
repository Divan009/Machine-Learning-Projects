# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 21:44:33 2018

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('banking.csv.txt', header=0)

data['education'].unique()
data['education']=np.where(data['education']=='basic.4y','Basic',data['education'])
data['education']=np.where(data['education']=='basic.6y','Basic',data['education'])
data['education']=np.where(data['education']=='basic.9y','Basic',data['education'])

#DAta Exploration
data['y'].value_counts()
sns.countplot(x='y',data=data,palette='hls')
plt.savefig('count_plot')

count_nosub=len(data[data['y']==0])
count_sub=len(data[data['y']==1])
tot_sub = count_nosub+count_sub
pct_nosub = count_nosub/tot_sub
pct_sub = count_sub/tot_sub
print("percentage of no subscription = ",pct_nosub*100)
print("percentage of subscription = ",pct_sub*100)

#classes are imbalanced
#continuous var
data.groupby('y').mean()
#categorical var
data.groupby('job').mean()
data.groupby('marital').mean()
data.groupby('education').mean()

#viz
pd.crosstab(data.job, data.y).plot(kind='bar')
plt.title('Purchase frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency for purchase')

pd.crosstab(data.education, data.y).plot(kind='bar')
plt.title('Purchase frequency for levels of education')
plt.xlabel('education')
plt.ylabel('Frequency for purchase')

pd.crosstab(data.marital, data.y).plot(kind='bar')
plt.title('Purchase frequency for Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Frequency for purchase')

pd.crosstab(data.day_of_week, data.y).plot(kind='bar')
plt.title('Purchase frequency for Day of week')
plt.xlabel('Day of week')
plt.ylabel('Frequency for purchase')


pd.crosstab(data.month, data.y).plot(kind='bar')
plt.title('Purchase frequency for month')
plt.xlabel('month')
plt.ylabel('Frequency for purchase')
 

pd.crosstab(data.poutcome, data.y).plot(kind='bar')
plt.title('Purchase frequency for poutcome')
plt.xlabel('poutcome')
plt.ylabel('Frequency for purchase')





































