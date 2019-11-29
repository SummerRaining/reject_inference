#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:33:04 2019

@author: tunan
"""

import os,json,pickle
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingClassifier,ExtraTreesClassifier,AdaBoostClassifier,RandomForestClassifier
from dataset import dataset
from util import print_analyse,plot_four_roc,find_best_threshold

class ml_model(object):
    '''
    base_model(estimator):定义使用什么模型
    adj_params:需要调整的参数
    params_dict:
    '''
    def __init__(self,base_model,adj_params,params_dict,int_feature,name):
        self.feature = None
        self.label = None
        self.model = None
        self.base_model = base_model   #未拟合前的模型。
        
        self.adj_params = adj_params
        self.params_dict = params_dict    #
        self.log_path = "../log/{}_config.json".format(name)   #最优参数的存储地址。
        self.model_path = "../models/{}_model".format(name)    #模型地址
        self.name = name
        self.int_feature = int_feature  #model中整数参数。
        
        if not os.path.exists('../log'):
            os.makedirs('../log')
            os.makedirs('../models')
            
    def load_model_with_params(self):
        #读取文件中的最优参数，并返回设置好参数的模型
        if not os.path.exists(self.log_path):
            raise ValueError("no file in {}".format(self.log_path))
            return None
        self.result = json.load(open(self.log_path,'r'))
        model = self.base_model(**self.result['params'])
        return model
            
    def auc_evaluate(self,**kwarg):
        '''
        description：给定参数下，模型的得分。
        inputs：
            kwarg(dict):需要调整的参数与其对应值。
        '''
        #固定的参数,与需调整的参数更新。得到模型完整参数params_dict.
        params_dict = self.params_dict  
        params_dict.update(kwarg)
        
        #将参数中整数型参数转换为整数。
        for f in self.int_feature:
            params_dict[f] = int(params_dict[f])
        
        #5折交叉验证，返回该参数下的模型评分。
        kf = KFold(5,shuffle=True,random_state=42)
        score = cross_val_score(self.base_model(**params_dict),
                                self.feature,self.label,
                                scoring = 'roc_auc',cv = kf)
        return score.mean()
    
    def select_params(self):
        '''
        description:贝叶斯优化得到最优的参数。与固定参数合并后，将整数型参数转换并储存下来。
        '''
        bayes = BayesianOptimization(self.auc_evaluate,self.adj_params)
        bayes.maximize()
        result = bayes.max
        
        #与固定参数合并
        best_params = self.params_dict
        best_params.update(result['params'])
        
        #转换其中的整数型参数。
        int_feature = self.int_feature
        for f in int_feature:
            best_params[f] = int(best_params[f])
        result['params'] = best_params
        
        #最优参数存到log路径中。
        self.result = result
        with open(self.log_path,'w') as f:
            f.write(json.dumps(result))
            
    def fit(self,X_train,y_train):
        ''' 
        description:
            1. 传入特征和标签，使用贝叶斯优化得到最优参数.
            2. 最有参数的模型拟合特征和标签训练最优模型，并存入文件。
            3. 清空特征和标签。
            4. 如果已有模型文件就直接读出。
        '''
        
        if not os.path.exists(self.model_path):
            self.feature = X_train
            self.label = y_train
            print('start training {} model!'.format(self.name))
            self.select_params()
            model = self.base_model(**self.result['params'])
            model.fit(X_train,y_train)
            pickle.dump(model,open(self.model_path,'wb'))
            self.feature = None
            self.label = None
        else:
            print("loading {} model from file".format(self.name))
            model = pickle.load(open(self.model_path,'rb'))
        self.model = model

    def predict_proba(self,X_test):
        return self.model.predict_proba(X_test)
    
    def _print_analyse(self,x_test,y_test,save_img = False):
        y_pred = self.predict_proba(x_test)
        print_analyse(y_test,y_pred[:,1],name = self.name) #打印分析模型性能的图形,后续定义。
        
class ensemble_model(object):
    def __init__(self,models,name):
        self.models = models
        self.name = name
    
    def predict_proba(self,X_test):
        pred = []
        for model in self.models:
            prob = model.predict_proba(X_test)
            #预测值为两列的情况
            if prob.ndim>1:
                prob = prob[:,1]
            pred.append(prob.reshape(-1,1))
        
        pred = np.concatenate(pred,axis = -1)
        return np.mean(pred,axis = -1)
            
    def _print_analyse(self,x_test,y_test):
        y_pred = self.predict_proba(x_test)
        print_analyse(y_test,y_pred,name = self.name)

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    '''
    定义了stacking模型。
    inputs: 
        base_models(list):确定好参数的基模型列表。
        meta_model(estimator):定义好参数的元模型。
    '''
    def __init__(self, base_models, meta_model,name, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        
        self.name = name
        self.model_path = "../models/{}_model".format(name)
        
    #重新定义fit和predict两个函数就可以完成一个学习器了
    def fit(self, X, y):
        '''
        1. 拟合模型，使用kfold.split(X,y)产生每轮交叉验证时的训练集和测试集。
        2. 每个模型都在各轮训练集上训练后在测试集上预测，循环结束后产生了n_sample*n_models大小的数据集。并保存拟合的模型。
        3. 使用产生的数据集训练元学习器。
        '''
        if not os.path.exists(self.model_path):
            print("start fitting stacking model!")
            self.base_models_ = [list() for x in self.base_models] #每个元素用于保存5个同种模型
            self.meta_model_ = clone(self.meta_model)
            kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
            
            # Train cloned base models then create out-of-fold predictions that are needed to train the cloned meta-model
            out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
            for i, model in enumerate(self.base_models):
                for train_index, holdout_index in kfold.split(X, y):
                    instance = clone(model)
                    self.base_models_[i].append(instance)
                    instance.fit(X[train_index], y[train_index])
                    y_pred = instance.predict(X[holdout_index])
                    out_of_fold_predictions[holdout_index, i] = y_pred
                    
            # Now train the cloned  meta-model using the out-of-fold predictions as new feature
            self.meta_model_.fit(out_of_fold_predictions, y)
            pickle.dump([self.base_models_,self.meta_model_],open(self.model_path,'wb'))
        else:
            print("load stacking model from {}".format(self.model_path))
            self.base_models_,self.meta_model_ = pickle.load(open(self.model_path,'rb'))

        return self
   
    def predict_proba(self, X):
        '''
        3种共15个模型对X分别预测，每种模型的预测值求平均得到新数据集。元模型对新数据集做预测得到预测值。
        '''
        meta_features = np.column_stack([
            np.column_stack([model.predict_proba(X)[:,1] for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict_proba(meta_features)
    
    def _print_analyse(self,x_test,y_test):
        y_pred = self.predict_proba(x_test)
        print_analyse(y_test,y_pred[:,1],name = self.name)
    

if __name__ == '__main__':
    #获取数据，训练数据和验证数据。
    data = dataset()
    accept_sample,validation,reject_sample = data.run()
    
    #缺失值填充
    accept_sample.fillna(0,inplace = True)
    validation.fillna(0,inplace = True)
    reject_sample.fillna(0,inplace = True)
    
    X_train = accept_sample.values[:,2:]
    y_train = accept_sample.values[:,0]
    X_test = validation.values[:,2:]
    y_test = validation.values[:,0]
    
    print("X_train shape is {}".format(X_train.shape))
    print("y_train shape is {}".format(y_train.shape))
    print("X_test shape is {}".format(X_test.shape))
    print("y_test shape is {}".format(y_test.shape))
        
    #lightgbm
    adj_dict = {'max_depth':(5,15),'n_estimators':(50,500),'learning_rate':(0.001,0.1),
                'num_leaves':(32,512),'min_child_samples':(20,100),'min_child_weight':(0.001,0.1),
                'feature_fraction':(0.5,1),'bagging_fraction':(0.5,1),'reg_alpha':(0,0.5),'reg_lambda':(0,0.5)}
    params_dict = {'objective':'binary','max_bin':200,'verbose':1,'metric':['auc','binary_logloss']}
    int_feature = ['n_estimators','max_depth','num_leaves','min_child_samples']
    light_model = ml_model(LGBMClassifier,adj_dict,params_dict,int_feature = int_feature,name = 'lightgbm')
    light_model.fit(X_train,y_train)
    light_model._print_analyse(X_test,y_test,save_img = True)
        
    #adaboost
    adj_dict = {"learning_rate":(0.001,0.3),'n_estimators':(50,500)}
    params_dict = {"algorithm":"SAMME.R"}
    int_feature = ["n_estimators"]
    ada_model = ml_model(AdaBoostClassifier,adj_dict,params_dict,int_feature=int_feature,name = 'adaboost')
    ada_model.fit(X_train,y_train)
    ada_model._print_analyse(X_test,y_test)
    
    #gbdt
    adj_dict = {'max_depth':(5,15),'min_samples_split':(0.0001,0.01),
                'subsample':(0.5,1),'learning_rate':(0.0001,0.1),'n_estimators':(50,500)}
    params_dict = {'random_state':1,'max_features':'sqrt','verbose':0}
    int_feature = ['max_depth','n_estimators']
    gbdt_model = ml_model(GradientBoostingClassifier,adj_dict,params_dict,int_feature = int_feature,name = 'gbdt')
    gbdt_model.fit(X_train,y_train)
    gbdt_model._print_analyse(X_test,y_test)
        
    #random forest
    adj_dict = {"max_depth":(5,11),'n_estimators':(50,500)}
    params_dict = {'verbose':0}
    int_feature = ['max_depth','n_estimators']
    rf_model = ml_model(RandomForestClassifier,adj_dict,params_dict,int_feature = int_feature,name = 'rf')
    rf_model.fit(X_train,y_train)
    rf_model._print_analyse(X_test,y_test)
    
    #xgboost
    adj_dict = {'n_estimators':(50,500),'max_depth':(5,20),'subsample':(0.5,1),
                'reg_alpha':(0.1,1),'reg_lambda':(0.1,1)}
    params_dict = {'learning_rate':0.1,  'min_child_weight':1, 'seed':0,
                   'colsample_bytree':0.8, 'gamma':0,'silent':1}
    int_feature = ['n_estimators','max_depth']
    xgb_model = ml_model(XGBClassifier,adj_dict,params_dict,int_feature = int_feature,name = 'xgboost')
    xgb_model.fit(X_train,y_train)
    xgb_model._print_analyse(X_test,y_test)
    
# =============================================================================
#     #集成
#     e_model = ensemble_model(models = [light_model,ada_model,gbdt_model,rf_model,xgb_model],name = 'ml model ensemble')
#     e_model._print_analyse(X_test,y_test)
# =============================================================================
    
    #stacking 提升法
    base_model = [gbdt_model.load_model_with_params(),rf_model.load_model_with_params(),xgb_model.load_model_with_params()]
    meta_model = make_pipeline(RobustScaler(),LogisticRegression(random_state=1,penalty = 'l2',solver = 'saga',max_iter = 100))
    stack_model = StackingAveragedModels(base_models=base_model,meta_model=meta_model,name = 'stacking')
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
                  name = "ROC curve based on the accepted")
    
    #预测样本
    threshold = find_best_threshold(y_test,stack_probas)
    reject_probas = stack_model.predict_proba(reject_sample.values[:,2:])[:,1]
    reject_predict = np.array(reject_probas>threshold,dtype = np.int32)
    pickle.dump(reject_predict,open("../intermediate/reject_predict",'wb'))
    print(np.sum(reject_predict))
    
# =============================================================================
#     import matplotlib
#     matplotlib.use("Qt5Agg")
#     import matplotlib.pyplot as plt
#     plt.hist(reject_probas,bins=30)
#     plt.title("stacking predict probabilitys!")
#     plt.show()
# =============================================================================
# =============================================================================
#     #extra_tree
#     adj_dict = {'max_depth':(5,50),'max_features':(0.5,1.0),'min_samples_leaf':(5,30),
#                 'min_samples_split':(10,70),'n_estimators':(50,500)}
#     params_dict = {'max_leaf_nodes':None,'min_impurity_decrease':0.0,
#                    'min_weight_fraction_leaf':0,'bootstrap':False,'criterion':'gini'}
#     int_feature = ['n_estimators','max_depth','min_samples_leaf','min_samples_split']
#     et_model = ml_model(ExtraTreesClassifier,adj_dict,params_dict,int_feature = int_feature,name = 'extra_tree')
#     et_model.fit(X_train,y_train)
#     et_model._print_analyse(X_test,y_test,save_img = True)
#     
# =============================================================================
# =============================================================================
#     #svm
#     adj_dict = {"C":(0.01,1000),'gamma':(0.001,1),'tol':(0.001,1.)}
#     params_dict = {'shrinking':True, 'kernel':'rbf','probability':True,'max_iter':-1}
#     svm_model = ml_model(SVC,adj_dict,params_dict,int_feature=[],name = 'svm')
#     svm_model.fit(X_train,y_train)
#     svm_model._print_analyse(X_test,y_test,save_img = True)
# =============================================================================
        