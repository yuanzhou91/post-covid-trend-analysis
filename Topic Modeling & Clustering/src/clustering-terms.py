#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 22:44:17 2020

@author: Yuan Zhou
"""
from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


FILE_NAME = str(sys.argv[1]) 
APPROACH_NAME = str(sys.argv[2]) 

def load_data (filename):
    return pd.read_json(filename, lines=True)


#### Extract file to dataframe, with terms as objects
df_org = load_data(FILE_NAME)
terms = df_org.columns

df = df_org.transpose()
X = df.values

items_df = load_data('./items.jl')
ids = items_df['id'].tolist()

df.columns = ids

##### Use K-Means to generate clusters
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


mds = TSNE(2,random_state=0)
X_2d = mds.fit_transform(X_scaled)


plt.rcParams['figure.figsize'] = [20, 20]
plt.rc('font', size=14)
plt.title("Terms|APPROACH_NAME={}".format(APPROACH_NAME))
for i in range(0, len(X_2d)):
    row = X_2d[i]
    plt.scatter(row[0],row[1], color='orange')
    plt.text(row[0], row[1], terms[i], fontsize=10)
plt.savefig("Terms|APPROACH_NAME={}.png".format(APPROACH_NAME), dpi=200)


hc = AgglomerativeClustering(n_clusters = None, distance_threshold=0, affinity = 'euclidean', linkage ='ward')
y_hc=hc.fit_predict(X_scaled)


import scipy.cluster.hierarchy as shc
plt.figure(figsize=(40, 20))  
plt.title("Dendrograms|APPROACH_NAME={}".format(APPROACH_NAME))
y_flat = np.reshape(y_hc, (len(y_hc), 1))
dend = shc.dendrogram(shc.linkage(y_flat, method='ward'), orientation='left', labels=terms)
plt.savefig("Dendrograms|APPROACH_NAME={}.png".format(APPROACH_NAME), dpi=100)

plt.close()
##### We got K==6 is the best clustering K.
K=5
kmeans = KMeans(n_clusters=K, random_state=0).fit(X_scaled)
classes = kmeans.labels_.tolist()

colors = ['red','green','blue','pink','orange', 'purple']
plt.rcParams['figure.figsize'] = [20, 20]
plt.rc('font', size=14)
plt.title("Terms Clustering|K={}|APPROACH_NAME={}".format(K, APPROACH_NAME))
for i in range(0, len(X_2d)):
    row = X_2d[i]
    x = row[0]
    y = row[1]
    plt.scatter(x,y,c=colors[classes[i]], label="Class {}".format(classes[i]))
    plt.text(x, y, i, fontsize=8)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.savefig("Terms|K={}|APPROACH_NAME={}.png".format(K, APPROACH_NAME), dpi=200)
