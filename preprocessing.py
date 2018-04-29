# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:56:57 2018

@author: Harsh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


dataset = pd.read_csv('cleaned_data.csv')

new_df = pd.concat([dataset.iloc[0], dataset.iloc[1]], axis = 0)

## Iterate through the dataset, find pairs and create new dataframe
## With partners
for index_1, row_1 in dataset.iterrows():
    for index_2, row_2 in dataset.iterrows():
        if(index_1 >= dataset.shape[0] or index_2 >= dataset.shape[0]):
            break
        if((dataset.iloc[index_1]['iid'] == dataset.iloc[index_2]['pid']) and (dataset.iloc[index_1]['pid'] == dataset.iloc[index_2]['iid'])):
            #print(index_1, index_2)
            new_row = pd.concat([dataset.iloc[index_1], dataset.iloc[index_2]], axis = 0)
            dataset.drop(index_1, inplace=True)
            dataset.drop(index_2, inplace=True)
            dataset = dataset.reset_index(drop=True)
            new_df = pd.concat([new_df, new_row], axis = 1)
            
    sys.stdout.write('.'); sys.stdout.flush();
    
    
    
new_df = new_df.T
new_df.to_csv('combined_data_no_repeats.csv')