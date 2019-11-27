#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:09:10 2019

@author: tunan
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns

trainFilePath = 'train/train_1.txt'
data = pd.read_csv(trainFilePath,sep = '\t')
print('default rate is : {:.5f}'.format(sum(data['label'])/len(data['label'])))
feature = data.iloc[:,4:]
label = data['label']

#select 2000 features most correlated with label
correlation  = []
for col in tqdm(feature.columns):
    df = pd.DataFrame({'feature':feature[col],'label':label})
    correlation.append(df.corr().iloc[0,1])
cor = pd.DataFrame({'name':feature.columns,'correlation':correlation})
cor.index = cor['name']
cor.drop('name',axis = 1,inplace=True)
cor[cor.isna().values] = 0
cor_abs = cor.apply(lambda x: abs(x))
cor_abs = cor_abs.sort_values('correlation',ascending = False)

print(cor_abs.index)
# =============================================================================
# #select 20 feature in columns
# sfeature = cor_abs.index[:20]
# 
# correlation = feature[sfeature].corr()
# sns.heatmap(correlation)
# 
# #select 20 feature in columns
# sfeature = cor_abs.index[:10]
# 
# correlation = feature[sfeature].corr()
# sns.heatmap(correlation)
# 
# #select 20 feature in columns
# sfeature = cor_abs.index[:30]
# 
# correlation = feature[sfeature].corr()
# sns.heatmap(correlation)
# 
# =============================================================================
