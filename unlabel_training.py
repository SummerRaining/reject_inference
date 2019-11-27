#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:49:15 2019

@author: tunan
"""

from main_func import *
train_x = pd.read_csv('train_data.csv')
train_y = train_x['label']
train_x = train_x.iloc[:,1:-1]

test_x = pd.read_csv('test_data.csv')
test_y = test_x['label']
test_x = test_x.iloc[:,1:-1]

# =============================================================================
# unlabeled_x = pd.read_csv('unlabeled_feature.csv')
# unlabeled_y = unlabeled_x['label']
# unlabeled_x = unlabeled_x.iloc[:,1:-1]
# 
# #set threshold let default rate equal to train_y
# default_rate = sum(train_y)/len(train_y)
# print("default rate in training set is {:.4f}".format(default_rate))
# threshold = 0
# for i in np.arange(0.01,1,0.001):
#     rate = sum(unlabeled_y>i)/len(unlabeled_y)
#     if(rate<default_rate):
#         print(f'threshold is {i} and rate is {rate}')
#         threshold = i
#         break
# new_y = (unlabeled_y>threshold).apply(int)
# 
# #merge train data and unlabeled data to build new train data
# train_x = pd.concat([train_x,unlabeled_x],axis = 0)
# train_y = pd.concat([train_y,new_y],axis = 0)
# del unlabeled_x
# =============================================================================

# =============================================================================
# stacked_averaged_models = StackingAveragedModels(base_models = (xgb_model, GBoost, rf_model),
#                                              meta_model = lr_model)
# stacked_averaged_models.fit(train_x.values,train_y.values)
# predict_y = stacked_averaged_models.predict(test_x.values)
# print("\nStacked averaged model score: {:.4f} \n".format(roc_auc_score(test_y,predict_y)))
# =============================================================================
def rmsle_cv(model,n_folds = 5):
    kf = KFold(n_folds,shuffle = True,random_state=42)
    rmsle = cross_val_score(model,feature.values,label.values,scoring = 'roc_auc',cv = kf)
    return(rmsle)
    
global feature,label
feature = pd.concat([train_x,test_x],axis = 0)
label = pd.concat([train_y,test_y],axis = 0)
stacked_averaged_models = StackingAveragedModels(base_models = (xgb_model, GBoost, rf_model),
                                              meta_model = lr_model)
score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

