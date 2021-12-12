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
terms = df.columns
items_df = load_data('./items.jl')
ids = items_df['id'].tolist()

##### Use K-Means to generate clusters
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=K, random_state=0).fit(X_scaled)

classes = kmeans.labels_.tolist()


mds = TSNE(2,random_state=0)
X_2d = mds.fit_transform(X_scaled)


colors = ['red','green','blue','cyan','orange', 'purple', 'pink', 'brown', 'grey', 'crimson']
plt.rcParams['figure.figsize'] = [20, 20]
plt.rc('font', size=14)
plt.title("Docs Clustering|K={}|APPROACH_NAME={}".format(K, APPROACH_NAME))
for i in range(0, len(X_2d)):
    row = X_2d[i]
    x = row[0]
    y = row[1]
    plt.scatter(x,y,c=colors[classes[i]], label="Class {}".format(classes[i]))
    plt.text(x, y, i, fontsize=8)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.savefig("Docs|K={}|APPROACH_NAME={}.png".format(K, APPROACH_NAME), dpi=200)


## Show summary of docs of each cluster
if K == 5:
    print("---------------------{}--------------------".format(APPROACH_NAME))
    for k in range(0, K):
        terms_weights = [0] * len(terms)
        for i, doc in enumerate(X_scaled):
            if classes[i] == k:
                for index in range(0, len(terms_weights)):
                    terms_weights[index] += doc[index]
        print( "Topic {} \n  Terms: {}".format( k, [x for _,x in sorted(zip(terms_weights, terms.copy()), reverse=True)][:10]))