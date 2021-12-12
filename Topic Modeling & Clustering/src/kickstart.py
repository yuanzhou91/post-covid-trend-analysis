#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 22:51:40 2020

@author: Yuan Zhou
"""
import os  # operating system commands

K = [2, 3, 4, 5, 6, 7, 8, 9, 10]

file_names = ['aj_df.jl', 'tfidf_df.jl','doc2vec_df.jl']

for k in K:    
    for file_name in file_names:        
        os.system("python ./clustering-docs.py {} {} {}".format(k, file_name, file_name.split('.')[0].split('_')[0]))
        
        
terms_file_names = ['aj_df.jl', 'tfidf_df.jl']
for file_name in terms_file_names:        
    os.system("python ./clustering-terms.py {} {}".format(file_name, file_name.split('.')[0].split('_')[0]))

os.system("python ./lda-topic-modeling.py")

os.system("python ./biclustering.py")
