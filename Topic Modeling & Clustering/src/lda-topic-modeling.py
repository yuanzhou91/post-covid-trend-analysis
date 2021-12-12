#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 01:00:01 2020

@author: Yuan Zhou
"""
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import LatentDirichletAllocation

#FILE_NAME = str(sys.argv[1]) 

FILE_NAME = 'aj_df.jl'

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic ", topic_idx)
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

def load_data (filename):
    return pd.read_json(filename, lines=True)


#### Extract file to dataframe
df = load_data(FILE_NAME)
X = df.values
feature_names = df.columns

items_df = load_data('./items.jl')
ids = items_df['id'].tolist()

##### Use K-Means to generate clusters
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

no_topics = 5
no_top_words = 10


# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=20).fit(X_scaled)

display_topics(lda, feature_names, no_top_words)
