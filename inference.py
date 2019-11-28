#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:00:00 2019

@author: tunan
"""
import os,json,pickle
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingClassifier,ExtraTreesClassifier,AdaBoostClassifier,RandomForestClassifier

from util import print_analyse,plot_four_roc
from dataset import dataset
from ml_model import StackingAveragedModels

class fit_ml_model(object):
    #这个类只拟合模型，不寻找最优参数。
    def __init__(self,base_model,name):
        self.feature = None
        self.label = None
        self.model = None
        self.base_model = base_model   #未拟合前的模型。
        
        self.log_path = "../log/{}_config.json".format(name)   #最优参数的存储地址。
        self.model_path = "../reject_models/{}_model".format(name)    #模型地址修改，reject_models
        self.name = name
        
        if not os.path.exists("../reject_models"):
            os.mkdir("../reject_models")
            
    def load_model_with_params(self):
        #读取文件中的最优参数，并返回设置好参数的模型
        if not os.path.exists(self.log_path):
            raise ValueError("no file in {}".format(self.log_path))
            return None
        self.result = json.load(open(self.log_path,'r'))
        model = self.base_model(**self.result['params'])
        return model
            
    def fit(self,X_train,y_train):
        ''' 
        description:
            1. 从文件中得到最优参数.
            2. 最优参数的模型拟合特征和标签训练最优模型，并存入文件。
            3. 清空特征和标签。
            4. 如果已有模型文件就直接读出。
        '''
        if not os.path.exists(self.model_path):
            self.feature = X_train
            self.label = y_train
            print('start training model!')
            self.load_model_with_params()
            model = self.base_model(**self.result['params'])
            model.fit(X_train,y_train)
            pickle.dump(model,open(self.model_path,'wb'))
            self.feature = None
            self.label = None
        else:
            print("loading model from file")
            model = pickle.load(open(self.model_path,'rb'))
        self.model = model

    def predict_proba(self,X_test):
        return self.model.predict_proba(X_test)
    
    def _print_analyse(self,x_test,y_test,save_img = False):
        y_pred = self.predict_proba(x_test)
        print_analyse(y_test,y_pred[:,1],name = "reject_"+self.name) 
            
if __name__ == "__main__":
    reject_predict = pickle.load(open("../intermediate/reject_predict",'rb'))
    data = dataset()
    accept_sample,validation,reject_sample = data.run()
    
    #缺失值填充
    accept_sample.fillna(0,inplace = True)
    validation.fillna(0,inplace = True)
    reject_sample.fillna(0,inplace = True)
    
    X_train = np.concatenate((accept_sample.values[:,2:],reject_sample.values[:,2:]),axis = 0)
    y_train = np.concatenate((accept_sample.values[:,0],reject_predict),axis = 0)
    
    X_test = validation.values[:,2:]
    y_test = validation.values[:,0]
    
    print("X_train shape is {}".format(X_train.shape))
    print("y_train shape is {}".format(y_train.shape))
    print("X_test shape is {}".format(X_test.shape))
    print("y_test shape is {}".format(y_test.shape))
    
    #############  模型拟合
    #lightgbm
    light_model = fit_ml_model(LGBMClassifier,name = 'lightgbm')
    light_model.fit(X_train,y_train)
    light_model._print_analyse(X_test,y_test,save_img = True)
        
    #adaboost
    ada_model = fit_ml_model(AdaBoostClassifier,name = 'adaboost')
    ada_model.fit(X_train,y_train)
    ada_model._print_analyse(X_test,y_test)
    
    #gbdt
    gbdt_model = fit_ml_model(GradientBoostingClassifier,name = 'gbdt')
    gbdt_model.fit(X_train,y_train)
    gbdt_model._print_analyse(X_test,y_test)
        
    #random forest
    rf_model = fit_ml_model(RandomForestClassifier,name = 'rf')
    rf_model.fit(X_train,y_train)
    rf_model._print_analyse(X_test,y_test)
    
    #xgboost
    xgb_model = fit_ml_model(XGBClassifier,name = 'xgboost')
    xgb_model.fit(X_train,y_train)
    xgb_model._print_analyse(X_test,y_test)
        
    #stacking 提升法
    base_model = [gbdt_model.load_model_with_params(),rf_model.load_model_with_params(),xgb_model.load_model_with_params()]
    meta_model = make_pipeline(RobustScaler(),LogisticRegression(random_state=1,penalty = 'l2',solver = 'saga',max_iter = 100))
    stack_model = StackingAveragedModels(base_models=base_model,meta_model=meta_model,name = 'reject_stacking')
    stack_model.fit(X_train,y_train)
    stack_model._print_analyse(X_test,y_test)
    
    #四种模型的图像画在一起
    rf_probas = rf_model.predict_proba(X_test)[:,1]
    xgb_probas = xgb_model.predict_proba(X_test)[:,1]
    gbdt_probas = gbdt_model.predict_proba(X_test)[:,1]
    ada_probas = ada_model.predict_proba(X_test)[:,1]
    stack_probas = stack_model.predict_proba(X_test)[:,1]
    plot_four_roc(y_test,[rf_probas,xgb_probas,ada_probas,stack_probas],\
                  model_names = ['random forest','xgboost','gbdt','stacking model'],\
                  name = "ROC curve based on the rejected")