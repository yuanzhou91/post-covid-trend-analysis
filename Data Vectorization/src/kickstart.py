#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 02:16:26 2020

@author: Yuan Zhou
"""
import os  # operating system commands
os.system('pip install -r ./requirements.txt')


vector_dimensions = [50, 100, 200, 500, 700, 1000]

os.system('python ./generate_labels.py')
os.system('python ./group_classes.py')

for vector_dimension in vector_dimensions:    
    os.system("python ./run_classifications.py {}".format(vector_dimension))