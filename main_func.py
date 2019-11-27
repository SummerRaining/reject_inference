#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:45:34 2019

@author: tunan
"""
# 训练集。id：样本编号1-100000，loan_dt：放款日期，label：逾期标签（1为逾期，0为非逾期，空字符串为未给出标签），
#tag：标识通过和拒绝用户（0为模型分数前30%-假设为通过，1为模型分数后70%-假设为拒绝），f1~f6745为特征。原文件过大，已拆分成多个小文件。
import os
import numpy as np
import pandas as pd
from tqdm import tqdm 

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from dataset import get_data


def rmsle_cv(model,n_folds = 5):
    '''
    输入调好参数的模型，返回模型使用交叉验证后的拟合的结果。
    description:cross_val_score以模型，特征和标签为输入，使用5折交叉验证。计算模型的评估结果。
    '''
    kf = KFold(n_folds,shuffle = True,random_state=42)
    rmsle = cross_val_score(model,feature.values,label.values,scoring = 'roc_auc',cv = kf)
    return(rmsle)
    

 

#define a model class with fit and predict function
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    '''
    定义了集成模型(取平均类)。
    inputs：models(list)，确定好参数的模型列表。
    outputs: self(obj),返回类的对象。对象中包括训练好的模型列表。
    '''
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict_proba(X)[:,1] for model in self.models_])
        print(f'shape is {predictions.shape}')
        return np.mean(predictions, axis=1)   


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    '''
    定义了stacking模型。
    inputs: 
        base_models(list):确定好参数的基模型列表。
        meta_model(estimator):定义好参数的元模型。
    outputs:
        
    '''
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    #重新定义fit和predict两个函数就可以完成一个学习器了
    def fit(self, X, y):
        '''
        拟合模型，是使用kfold.split(X,y)产生每轮交叉验证时的训练集和测试集。
        每个模型都在各轮训练集上训练后在测试集上预测，循环结束后产生老n_sample*n_models大小的数据集。并保存拟合的模型。
        使用产生的数据集训练元学习器。
        '''
        self.base_models_ = [list() for x in self.base_models]
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
        return self
   
    def predict(self, X):
        '''
        3种共15个模型对X分别预测，每种模型的预测值求平均得到新数据集。元模型对新数据集做预测得到预测值。
        '''
        meta_features = np.column_stack([
            np.column_stack([model.predict_proba(X)[:,1] for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict_proba(meta_features)[:,1]
    
if __name__ == '__main__':
    #定义全局变量特征和标签。
    global feature,label
    #得到特征，标签和拒绝样本的特征。
    feature,label,unlabeled_feature = get_data()
    
    #已经调好的xgboost参数。传入参数使用列表加上××的方法。
    params_dist = {'learning_rate': 0.1, 'n_estimators': 169, 'max_depth': 10, 'min_child_weight': 1,
                   'seed': 0,'subsample': 0.9413, 'colsample_bytree': 0.8, 
                   'gamma': 0,'silent':1, 'reg_alpha': 0.2357,
                   'reg_lambda': 0.1538,'tree_method': 'gpu_hist'}
    #使用固定的参数生成一个xgb模型。
    xgb_model = XGBClassifier(**params_dist)
    #使用管道做标准化，减去中位数除四分位距。预处理后的输出作为逻辑回归的输入。
    lr_model = make_pipeline(RobustScaler(),LogisticRegression(random_state=1,penalty = 'l2',solver = 'saga',max_iter = 100))
    #使用设置好的值得到Gboost和rf.
    GBoost = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                        max_depth=4, max_features='sqrt',
                                        min_samples_leaf=15, min_samples_split=10, 
                                        loss='deviance', random_state =5)
    rf_model = RandomForestClassifier(n_estimators=100,max_depth = 5)
        
    #对定义好的xgboost，测试交叉验证的auc值。
    score = rmsle_cv(xgb_model)
    print("\nxgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    
    #对定义好的xgboost，测试交叉验证的auc值。
    score = rmsle_cv(GBoost)
    print("\nGradient descent boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    
    #对定义好的rf，测试交叉验证的auc值。
    score = rmsle_cv(rf_model)
    print("\nRandom forest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    
    #对定义好的平均类对象，测试交叉验证的auc值。
    averaged_models = AveragingModels(models = (xgb_model, GBoost, rf_model))
    score = rmsle_cv(averaged_models)
    print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    
    #对定义好的stacking类对象，测试交叉验证的auc值。
    stacked_averaged_models = StackingAveragedModels(base_models = (xgb_model, GBoost, rf_model),
                                                 meta_model = lr_model)
    score = rmsle_cv(stacked_averaged_models)
    print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    
    
    #定义stacking类的对象，将接受样本分割训练集和验证集，训练完模型后显示在验证集上的结果。
    stacked_averaged_models = StackingAveragedModels(base_models = (xgb_model, GBoost, rf_model),
                                                 meta_model = lr_model)
    X_train, X_test, y_train, y_test = train_test_split(feature,label, test_size=0.1, random_state=42)
    stacked_averaged_models.fit(X_train.values,y_train.values)
    unlabeled_y = stacked_averaged_models.predict(unlabeled_feature.values)
    
    #stacking类的对象，对拒绝样本做预测，使用0.12作为阈值。导出加上预测值的拒绝样本。
    #注意这里有一个bug，有标签的拒绝样本也被放入这里面了。
    #这是由于接受样本的违约率是0.06，所以我们认为拒绝样本的违约是其一倍。设为0.12
    print('defalt rate is {:.4f}'.format(sum(unlabeled_y>0.5)/len(unlabeled_y)))
    unlabeled_feature['label'] = unlabeled_y
    unlabeled_feature.to_csv('unlabeled_feature.csv')
    
    #导出分隔后的接受样本。
    X_train['label'] = y_train.values
    X_test['label'] = y_test.values
    X_train.to_csv('train_data.csv')
    X_test.to_csv('test_data.csv')
    
