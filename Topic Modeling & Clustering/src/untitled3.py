#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 14:45:48 2020

@author: yluo
"""


import numpy as np
import pandas as pd

def load_data (filename):
    return pd.read_json(filename, lines=True)

df = load_data('aj_df.jl')
X = df.values
terms = df.columns


for t in terms:
    print(t)
