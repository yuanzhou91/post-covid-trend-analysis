#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 02:16:26 2020

@author: Yuan Zhou
"""
import pandas as pd
import numpy as np

def load_data (filename):
    return pd.read_json(filename, lines=True)

df = load_data('items_with_labels.jl')

target_classes = {'Computers', 'Health', 'Business'}

new_targets = []
for target in df['topic']:
    if target in target_classes:
        new_targets.append(target)
    else:
        new_targets.append('Other')
    
df['topic'] = new_targets
    
classes_dict = {'Computers': 0, 'Health': 1, 'Business': 2, 'Other': 3}

classes_counts = [0, 0, 0, 0]

for target in df['topic']:
    idx = classes_dict[target]
    classes_counts[idx] = classes_counts[idx] + 1
    
for target_class in classes_dict.keys():
    print('Class: ', target_class, ', count:', classes_counts[classes_dict[target_class]])


file_with_labels = open("items_with_grouped_labels.jl", "w")
df.to_json("items_with_grouped_labels.jl",
           orient="records",
           lines=True)
