# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:14:29 2019

@author: Ganesh
"""

import numpy as np #for mathematical calculations
import pandas as pd #for reading a different datasets
import matplotlib.pyplot as plt #for visualizations

calories = pd.read_csv("F:\\R\\files\\calories_consumed.csv")

calories = calories.rename(columns = {"Weight gained (grams)": "weight_gained", "Calories Consumed": "calories_consumed"} )

calories.columns

plt.plot(calories.calories_consumed)
plt.plot(calories.calories_consumed, calories.weight_gained)

calories.corr() # high correlation between them

#build a model

#model1 
import statsmodels.formula.api as smf

model1 = smf.ols("weight_gained~calories_consumed", data = calories).fit()

model1.summary()

model1.conf_int(0.05)
pred1 = model1.predict(calories)
pred1.corr(calories.iloc[:,1]) #100% correlation between them

finalmodel = model1

residuals = model1.resid_pearson
residuals
