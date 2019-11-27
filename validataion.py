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
from sklearn.model_selection import train_test_split 
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from dataset import get_data
import json
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix,roc_curve, auc
import matplotlib.pyplot as plt 

#定義stacking模型。
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds   
    # We again fit the data on clones of the original models
    #重新定义fit和predict两个函数就可以完成一个学习器了
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
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
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict_proba(X)[:,1] for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict_proba(meta_features)[:,1] 


def get_xgbmodel():
    '''
    读取xgboost的最优参数，返回最优参数下的xgboost模型对象。
    '''
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
    xgb_model = XGBClassifier(**best_params)
    return xgb_model

def get_gbdtmodel():
    best_params = {}
    with open('gbdt_config.json','r') as f:
        best_params = json.loads(f.read())
    best_params = best_params['params']
    best_params['n_estimators'] = int(best_params['n_estimators'] )
    best_params['max_depth'] = int(best_params['max_depth'] )
    gbdt_model = GradientBoostingClassifier(**best_params)
    return gbdt_model    

def get_stackmodel():
    '''
    '''
    #使用管道将预处理和lr结合在一起，得到lr模型。
    lr_model = make_pipeline(RobustScaler(),
                             LogisticRegression(random_state=1,penalty = 'l2',solver = 'saga',max_iter = 100))
    
    #得到最优的xgboost模型
    xgb_model = get_xgbmodel()
    params = xgb_model.get_params()
    del params['tree_method']
    xgb_model = XGBClassifier(**params)
    
    #得到最优的gbdt,rf模型。
    GBoost = get_gbdtmodel()
    rf_model = RandomForestClassifier(n_estimators=100,max_depth = 5)
    #将以上的最优模型作为参数，得到stacking模型。
    stacked_averaged_models = StackingAveragedModels(base_models = (xgb_model, GBoost, rf_model),
                                                  meta_model = lr_model)
    return stacked_averaged_models
    

def plot_roc(ytrue,ypred,name):
    '''
    使用真实值和预测值，画出ROC曲线。
    '''
    #使用roc_curve，计算真阳率和假阳率
    fpr,tpr,threshold = roc_curve(ytrue,ypred ) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name+' roc curve')
    plt.legend(loc="lower right")
    
    path = 'roc image'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path,name+".png"))
    plt.show()
    

if __name__ == '__main__':
    #得到特征和标签。
    global feature,label
    feature,label,unlabel_feature = get_data()
    del unlabel_feature
    #split test data for testing and write train unlabeled data test data in disk
    
    #分割数据集
    X_train, X_test, y_train, y_test = train_test_split(feature,label, test_size=0.1, random_state=42)
    del feature,label
    
    #导出最优xgboost，在训练集上训练后，预测结果。绘制roc图形。
    xgb_model = get_xgbmodel()
    xgb_model.fit(X_train.values,y_train.values)
    predict_y = xgb_model.predict_proba(X_test.values)[:,1]
    plot_roc(y_test,predict_y,'xgboost model')
    
    #导出最优gbdt，在训练集上训练后，预测结果。绘制roc图形。
    gbdt_model = get_gbdtmodel()
    gbdt_model.fit(X_train,y_train.values)
    predict_y = gbdt_model.predict_proba(X_test.values)[:,1]
    plot_roc(y_test,predict_y,'gbdt model')
    
    #导出最优rf，在训练集上训练后，预测结果。绘制roc图形。
    rf_model = RandomForestClassifier(n_estimators=100,max_depth = 5)
    rf_model.fit(X_train,y_train.values)
    predict_y = rf_model.predict_proba(X_test.values)[:,1]
    plot_roc(y_test,predict_y,'rf model')
    
    #导出最优stacking模型，在训练集上训练后，预测结果。绘制roc图形。
    stack_model = get_stackmodel()
    stack_model.fit(X_train.values,y_train.values)
    predict_y =stack_model.predict(X_test.values)
#    predict_y =stack_model.predict_proba(X_test.values)[:,1]
    plot_roc(y_test,predict_y,'stack model')
    
    #通过改变阈值，得到最好的f1.
    f1max = 0
    thmax = 0
    for threshold in np.arange(0,0.5,0.0001):
        py = (predict_y>threshold)
        f1 = f1_score(y_test,py)
        if f1>f1max:
            thmax = threshold
            f1max = f1
            
    py = predict_y>thmax
    #混淆矩阵，计算最优f1下的混淆矩阵。和此时的precision,recall,accuracy。
    C = confusion_matrix(y_test,py)    
    confusion = pd.DataFrame(C,columns = ['predict_0','predict_1'],index = ['true_0','true_1'])
    print(confusion)
    
    precision = C[1,1]/(C[0,1]+C[1,1])
    recall = C[1,1]/(C[1,0]+C[1,1])
    F1 = 2*(precision*recall)/(precision+recall)
    accuracy = sum(py==y_test)/len(y_test)
    print("precision:{:.4f},recall:{:.4f},F1:{:.4f},accuracy:{:.4f}".format(precision,recall,F1,accuracy))
    

