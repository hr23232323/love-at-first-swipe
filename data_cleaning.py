# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 20:09:47 2018

@author: Harsh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

raw_data = pd.read_csv('Speed Dating Data.csv')

# Only needed columns
focus_data = raw_data.iloc[:, [0, 1, 2, 10, 11, 12, 13, 14, 15, 69, 70, 71, 72, 73, 74, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 97, 98, 99, 100, 101, 102, 103, 104, 105]]

#Replace nan with 0
focus_data = focus_data.fillna(0)

#export as csv
focus_data.to_csv('cleaned_data.csv')

