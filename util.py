#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:51:49 2019

@author: tunan
"""

from sklearn.metrics import confusion_matrix,roc_curve, auc,roc_auc_score
import matplotlib.pyplot as plt 
import os
import matplotlib
import numpy as np
matplotlib.use('Agg')
#matplotlib.use("Qt5Agg")

def print_analyse(ytrue,yproba,name):
    '''
    使用真实值和预测值，画出ROC曲线,auc值，一二类错误率等评估信息。
    inputs:
        ytrue,yproba(array)
    '''
    if(len(yproba.shape) != 1):
        raise ValueError("dimension of yproba should be one!")

    #计算auc的值
    roc_auc = roc_auc_score(ytrue,yproba)
    #计算混淆矩阵和第一二类错误率，准确率
    ypred = yproba>0.5
    con_max = confusion_matrix(ytrue,ypred)
    TN = con_max[0,0]
    TP = con_max[1,1]
    FN = con_max[1,0]
    FP = con_max[0,1]
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*recall*precision/(recall+precision)
    print("\n\n"+"*"*10+" {} ".format(name)+"*"*10)
    print("预测样本数为{}. \nAUC为{:.4f}".format(len(ytrue),roc_auc))
    print("准确率accuracy为{:.3f}%".format((con_max[1,1]+con_max[0,0])/len(ytrue)*100))
    print("第一类错误样本有{},第一类错误率为{:.3f}%".format(FN,FN/(FN+TP)*100))
    print("第二类错误样本有{},第二类错误率为{:.3f}%".format(FP,FP/(FP+TN)*100))
    print("精确率precisoin为{:.3f}%,召回率recall为{:.3f}%，F1为{:.3}% \n".format(recall*100,precision*100,F1*100))
    print("TP:{}\tTN:{},FN:{},FP:{}\n".format(TP,TN,FN,FP))
    
    #绘制roc曲线
    plot_roc(ytrue,yproba,name)
    
def plot_roc(ytrue,yproba,name):
    #使用roc_curve，计算真阳率和假阳率
    fpr,tpr,threshold = roc_curve(ytrue,yproba) 
    roc_auc = auc(fpr,tpr)
    
    plt.figure()    
    lw = 2  #线段的宽度
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name+' roc curve')
    plt.legend(loc="lower right")
    
    path = 'roc_image'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path,name+".png"))
#    plt.show()
    
def plot_four_roc(ytrue,yprobas,model_names,name = "four_model_roc_curve"):    
    plt.figure()    
    lw = 2  #线段的宽度
    plt.figure(figsize=(10,10))
    colors = ['cyan','yellow','red','blue']
    for yproba,model_name,color in zip(yprobas,model_names,colors):
        fpr,tpr,threshold = roc_curve(ytrue,yproba) 
        roc_auc = auc(fpr,tpr)
        plt.plot(fpr, tpr, color=color,
                 lw=lw, label='{} AUC : {:.3f}'.format(model_name,roc_auc)) ###假正率为横坐标，真正率为纵坐标做曲线

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right")
    
    path = 'roc_image'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path,name+".png"))
#    plt.show()