# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:32:26 2019

@author: lenovo
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
campaign = pd.read_csv('campaign_data.csv')

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

bow_vectorizer_uni = CountVectorizer(ngram_range=(1,1), stop_words='english')