#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 22:44:17 2020

@author: Yuan Zhou
"""
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


K = int(sys.argv[1])
FILE_NAME = str(sys.argv[2]) 
APPROACH_NAME = str(sys.argv[3]) 

def load_data (filename):
    return pd.read_json(filename, lines=True)


#### Extract file to dataframe
df = load_data(FILE_NAME)
X = df.values

items_df = load_data('./items.jl')
ids = items_df['id'].tolist()

##### Use K-Means to generate clusters
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=K, random_state=0).fit(X_scaled)

classes = kmeans.labels_.tolist()


mds = TSNE(2,random_state=0)
X_2d = mds.fit_transform(X_scaled)


colors = ['red','green','blue','black','orange', 'purple', 'pink', 'brown', 'grey', 'crimson']
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
for i in range(0, K):
    subset = []
    for idx in range(0, len(classes)):
        if classes[idx] == i:
            subset.append(X_2d[idx])
    x = [row[0] for row in subset]
    y = [row[1] for row in subset]
    plt.scatter(x,y,c=colors[i],label=i)
plt.legend()
plt.savefig("K={}|APPROACH_NAME={}.png".format(K, APPROACH_NAME))
plt.show()
