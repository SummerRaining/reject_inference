#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:32:09 2019

@author: tunan
"""

from bayes_opt import BayesianOptimization
import json
from main_func import *
import copy


'''
该文件用于对rf，xgboost，lightgbm等模型进行调参
'''
#得到数据特征和标签
feature,label,_ = get_data()

def auc_evaluate_xgb(n_estimators,max_depth,subsample,reg_alpha,reg_lambda):
    '''
    定义xgboost的参数函数。输入相关参数，输出该参数条件下交叉验证得到的结果AUC。
    inputs:
        n_estimators(float):基模型的个数。
        max_depth(float):最大深度
        subsample(float):子采样率
        reg_alpha,reg_lambda(float):正则化参数
    '''
    #无需调整的参数。
    params_dist = {'learning_rate': 0.1,  'min_child_weight': 1, 'seed': 0,
                    'colsample_bytree': 0.8, 'gamma': 0,
                    'silent':1,'tree_method': 'gpu_hist'} 
    #将部分参数转换成int型。
    params_dist['n_estimators'] = int(n_estimators)
    params_dist['max_depth'] = int(max_depth)
    params_dist['subsample'] = float(subsample)
    params_dist['reg_alpha'] = float(reg_alpha)
    params_dist['reg_lambda'] = float(reg_lambda)
    
    #使用交叉验证评分函数cross_val_score，得到5折交叉验证时的得分。
    kf = KFold(5,shuffle=True,random_state=42)
    score = cross_val_score(XGBClassifier(**params_dist),feature.values,label.values,
                            scoring='roc_auc',cv = kf)
    return score.mean()

def select_params_xgboost(adj_params,path = 'xgb_config.json'):
    '''
    bayes优化调参。得到的结果写入xgb_config.json文件中。
    '''
    bayes = BayesianOptimization(auc_evaluate_xgb,adj_params)
    bayes.maximize()
    result = bayes.max
    
    params_dist = {'learning_rate': 0.1,  'min_child_weight': 1, 'seed': 0,
                    'colsample_bytree': 0.8, 'gamma': 0,
                    'silent':1,'tree_method': 'gpu_hist'} 
        
    best_params = result['params']
    best_params['n_estimators'] = int(best_params['n_estimators'] )
    best_params['max_depth'] = int(best_params['max_depth'] )
    params_dist.update(best_params)
    result['params'] = params_dist
    
    with open(path,'w') as f:
        f.write(json.dumps(result))
        
def auc_evaluate_gbdt(max_depth,min_samples_split,subsample):   
    #set it as global value
    params_dist = copy.copy(gbdt_params_dict)
# =============================================================================
#     params_dist['n_estimators'] = int(n_estimators)
#     params_dist['learning_rate'] = 10**(-learning_rate)
# =============================================================================
    params_dist['max_depth'] = int(max_depth)
    params_dist['min_samples_split'] = float(min_samples_split)
    params_dist['subsample'] = float(subsample)
    #every stimulation we use 5-fold cross validation
    kf = KFold(5,shuffle=True,random_state=42)
    score = cross_val_score( GradientBoostingClassifier(**params_dist),feature.values,label.values,
                            scoring='roc_auc',cv = kf)
    return score.mean()

def select_params_gbdt(adj_params,inits = 5,n_iter=25,path = 'gbdt_config.json'):
    #同理
    bayes = BayesianOptimization(auc_evaluate_gbdt,adj_params)
    bayes.maximize(init_points=inits,n_iter=n_iter)
    result = bayes.max
    
    params_dict = copy.copy(gbdt_params_dict)
    best_params = result['params']
    params_dict.update(best_params)

    params_dict['n_estimators'] = int(params_dict['n_estimators'])
    params_dict['max_depth'] = int(params_dict['max_depth'] )
    result['params'] = params_dict
    
    with open(path,'w') as f:
        f.write(json.dumps(result))
        
        
def auc_evaluate_rf(n_estimators,max_depth,subsample,learning_rate):
    params_dist = {} 
    
    params_dist['n_estimators'] = int(n_estimators)
    params_dist['max_depth'] = int(max_depth)
    params_dist['subsample'] = float(subsample)
    
    #every stimulation we use 5-fold cross validation
    kf = KFold(5,shuffle=True,random_state=42)
    score = cross_val_score(RandomForestClassifier(**params_dist),feature.values,label.values,
                            scoring='roc_auc',cv = kf)
    return score.mean()
        
def select_params_rf(adj_params,path = 'rf_config.json'):
    #find best params and dump it to json text
    bayes = BayesianOptimization(auc_evaluate_rf,adj_params)
    bayes.maximize()
    result = bayes.max
    
    params_dist = {} 
    
    best_params = result['params']
    best_params['n_estimators'] = int(best_params['n_estimators'] )
    best_params['max_depth'] = int(best_params['max_depth'] )
    params_dist.update(best_params)
    result['params'] = params_dist
    
    with open(path,'w') as f:
        f.write(json.dumps(result))
        
        
        
if __name__ == '__main__':
    #定义xgboost的可调节参数，并写入文件中。 
    adj_params = {'n_estimators': (50,500), 
                  'max_depth': (5,10.99),
                  'subsample': (0.5,1),
                  'reg_alpha': (0.1,1),
                  'reg_lambda': (0.1,1)
                  }
    select_params_xgboost(adj_params,'xgb_config.json')
    
    #定义gbdt的可调节参数，并写入文件中。
    global gbdt_params_dict 
    gbdt_adj_params = { 'max_depth':(5,15),
                       'min_samples_split':(0.0001,0.01),
                       'subsample':(0.5,1)}
    gbdt_params_dict = {'learning_rate':0.06152,
                        'n_estimators':180,
                        'random_state':1,
                        'max_features':'sqrt',
                        'verbose':0} 
    select_params_gbdt(gbdt_adj_params,inits=3,n_iter=15)
    
    #定义rf的可调节参数，并写入文件中。
    rf_adj_dict = {"max_depth":(5,11),'n_estimators':(50,500)}
    select_params_rf(rf_adj_dict)
    
#定义模型的评价函数。返回模型5折交叉验证后的值。
def rmsle_cv(model,n_folds = 5):
    kf = KFold(n_folds,shuffle = True,random_state=42)
    rmsle = cross_val_score(model,feature.values,label.values,scoring = 'roc_auc',cv = kf)
    return(rmsle)
    
#取出xgboost的最优参数。
params_dist = {'learning_rate': 0.1,  'min_child_weight': 1, 'seed': 0,
                'colsample_bytree': 0.8, 'gamma': 0,
                'silent':1,'tree_method': 'gpu_hist'} 
best_params = {}
with open('config.json','r') as f:
    best_params = json.loads(f.read())
best_params = best_params['params']
best_params['n_estimators'] = int(best_params['n_estimators'] )
best_params['max_depth'] = int(best_params['max_depth'] )

best_params.update(params_dist)

#训练模型，显示最优得分。
xgb_model = XGBClassifier(**best_params)
score = rmsle_cv(xgb_model)
print("\nxgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))






