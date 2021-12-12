#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 22:45:15 2020

@author: Yuan Zhou
"""
import os
import pandas as pd
import numpy as np

##### Install necessary dependencies ######
os.system('pip install -r ./requirements.txt')



###### Initialize uclassify clients ########

from uclassify import uclassify

account = uclassify()
account.setReadApiKey('VYrUNzSc0J8i')


def load_data (filename):
    return pd.read_json(filename, lines=True)


def generate_labels(df):
    body_list = df['body'].tolist()
    raw_labels = account.classify(body_list,"topics", username='uclassify')
    labels_of_all_categories = list(map(lambda x: x[2], raw_labels))
    labels = []
    for labels_list in labels_of_all_categories:
        sorted_labels = sorted(labels_list, key=lambda x: float(x[1]), reverse=True)
        labels.append(sorted_labels[0][0])
    return labels

    
df = load_data('items.jl')
targets = generate_labels(df)

df['topic'] = targets

file_with_labels = open("items_with_labels.jl", "w")
print(df.columns)
df.to_json("items_with_labels.jl",
           orient="records",
           lines=True)
