#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:13:06 2018

@author: tunan
xgboost anaconda安装方法
conda install -c conda-forge xgboost=0.6a2
"""
# 训练集。id：样本编号1-100000，loan_dt：放款日期，label：逾期标签（1为逾期，0为非逾期，空字符串为未给出标签），
#tag：标识通过和拒绝用户（0为模型分数前30%-假设为通过，1为模型分数后70%-假设为拒绝），f1~f6745为特征。原文件过大，已拆分成多个小文件。
import os
import numpy as np
import pandas as pd
from tqdm import tqdm 
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split 
'''
先用一个数据文件进行预测，熟悉使用后使用5个数据文件,第一个文件夹中只有通过的文件。
xgboost有自带的缺失值处理方法，可以直接使用，train_1.txt中没有无标签的数据
'''
if __name__ == '__main__':
    trainFilePath = 'train/train_1.txt'
    data = pd.read_csv(trainFilePath,sep = '\t')
    print('default rate is : {:.5f}'.format(sum(data['label'])/len(data['label'])))
    feature = data.iloc[:,4:]
    label = data.iloc[:,3]
    sum(pd.isnull(label))
    sum(data['tag'])
    
    
    # 随机抽取20%的测试集
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.1)
    print(len(X_train), len(X_test))
    
    param_dist = {'objective':'binary:logistic', 'n_estimators':2}
    clf = XGBClassifier(**param_dist)
   
    clf.fit(X_train, y_train,
           eval_set=[(X_train, y_train), (X_test, y_test)],
           eval_metric='logloss',
           verbose=True)
   
    evals_result = clf.evals_result()
   
