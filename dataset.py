#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 22:11:11 2019

@author: tunan
"""
import numpy as np
import pandas as pd
from tqdm import tqdm 
import os,pickle
from sklearn.utils import shuffle
'''
dataset有两个目的：
    1.做特征筛选，只筛选2000个有用的特征用于预测。
    2.将5个数据集合并，并且分成接受样本，拒绝样本，有标签的拒绝样本。 
    3.对三种数据切分训练集，验证集。部分接受样本加上有标签的拒绝样本作为验证集，剩余接受样本作为训练集。
因此整合的函数应该返回:训练集和验证集。使用一个类的原因是，这些操作都属于对数据做预处理部分，功能相近并且存在调用关系。
设计思路：
    dataset定义成class。成员函数有: select_feature，combine_data，split_data。
使用类的优点：
    1. 程序设计时，会定义多个函数。往往有一个函数返回的结果是另一个函数的输入。
        这时需要在主函数里定义变量，在第二个函数中定义形参接受。变量名过多使用复杂。
    2. 对于函数间传输的变量，使用类可以定义成中间变量。每个成员函数都可以随时调用。
    3. 功能相近的函数作为类的成员函数，减缓命名冲突。
'''
class dataset(object):
    def __init__(self):
        #将class需要传递的变量初始化。
        self.path = '../inputs'
        self.feature = pd.read_csv(os.path.join(self.path,'train_1.txt'),sep = '\t').columns
        self.select_features = None        
        
        self.accept_sample = None
        self.reject_sample = None
        self.labeled_sample = None
        self.validation = None
        
        self.data_path = '../intermediate/washed_data.pkl'
        if not os.path.exists("../intermediate"):
            os.makedirs('../intermediate')
    
    def select_feature(self,data,target,N = 2000):
        '''
        select N features most correlated with target,and return features' name
        inputs:
            data(dataframe):包含多个特征的数据框。
            target(array):标签数组。
        return:
            index(serial):与target最相关的N个特征。
        '''
        correlation = []
        for col in tqdm(data.columns):
            df = pd.DataFrame({'feature':data[col],'label':target})
            correlation.append(df.corr().iloc[0,1])
        cor = pd.DataFrame({'name':data.columns,'cor':correlation})
        cor.index = cor['name']
        cor.drop('name',axis = 1,inplace = True)
        
        #根据相关系数的绝对值进行排序。
        cor[cor.isna().values] = 0
        cor_abs = cor.apply(lambda x: abs(x))
        cor_abs = cor_abs.sort_values('cor',ascending = False)
        return cor_abs[:N].index
    
    def combine_data(self):        
        #read data and split them into three part,accept_sample,reject_sample,labeled_sample
        inputs = []
        for index in tqdm(range(1,6)):
            path = os.path.join(self.path,'train_{}.txt'.format(index))
            if index == 1:
                data = pd.read_csv(path,sep = '\t')
            else:
                data = pd.read_csv(path,sep = '\t',names = self.feature)
            data = data[['label','tag'] + list(self.select_features)]
            inputs.append(data)
            
        inputs = pd.concat(inputs,axis = 0)
        #split inputs into three part
        self.accept_sample = inputs[inputs['tag'] == 0]
        self.reject_sample = inputs[(inputs['tag']==1) & pd.isna(inputs['label'])]
        self.labeled_sample = inputs[inputs['tag']==1 & (~pd.isna(inputs['label'])) ] 
        print("accept {},reject: {},labeled: {} sample".format(len(self.accept_sample),len(self.reject_sample),len(self.labeled_sample)))  
        
    def split_data(self):
        accept_sample,reject_sample,labeled_sample = self.accept_sample,self.reject_sample,self.labeled_sample
        
        #generate validation data
        accept_sample = shuffle(accept_sample,random_state = 2019)
        validation = pd.concat([accept_sample[-1300:],labeled_sample],axis = 0)
        accept_sample = accept_sample[:-1300]
        print("accept {},reject: {},validation: {} sample".format(len(accept_sample),len(reject_sample),len(validation)))  
       
        self.accept_sample,self.validation = accept_sample,validation
    
    def run(self):
        '''
        从路径中读取特征数据，或者特征处理产生数据。
        '''
        if os.path.exists(self.data_path):
            print("\n load data from {}".format(self.data_path))
            result = pickle.load(open(self.data_path,'rb'))
        else:
            #select feature 
            data = pd.read_csv(os.path.join(self.path,'train_1.txt'),sep = '\t')
            feature,label = data.iloc[:,4:],data['label'].values
            print("\n start selecting features!")
            self.select_features  = self.select_feature(feature,label,N=2000)
            #combine data
            print("\n start combing data!")
            self.combine_data()
            #split data
            print("\n start spliting data!")
            self.split_data()
            
            #将所有数据存入文件中。
            result = [self.accept_sample,self.validation,self.reject_sample]
            print("\n save data into {}".format(self.data_path))
            pickle.dump(result,open(self.data_path,'wb'))
            
        return result
        
if __name__ == '__main__':
    data = dataset()
    accept_sample,validation,reject_sample = data.run()
    X_train = accept_sample.values[:,2:]
    y_train = accept_sample.values[:,0]
    X_test = validation.values[:,2:]
    y_test = validation.values[:,0]
    
    print("X_train shape is {}".format(X_train.shape))
    print("y_train shape is {}".format(y_train.shape))
    print("X_test shape is {}".format(X_test.shape))
    print("y_test shape is {}".format(y_test.shape))
    
# =============================================================================
#     #这个里面的文件是后续添加的，用于读取文件并统计三个数据集中的违约率。
#     data1 = pd.read_csv('train/train_1.txt',sep = '\t')
#     columns = data1.columns
#     accept_data = {'label':[],'tag':[]}
#     reject_data = {'label':[],'tag':[]}
#     
#     for index in tqdm(range(1,6)):
#         path = 'train/train_{}.txt'.format(index)
#         if path == 'train/train_1.txt':
#             data = pd.read_csv(path,sep = '\t')
#         else:
#             data = pd.read_csv(path,sep = '\t',names = columns)
#             
#         acc_num = data['tag']==0
#         accept_data['label'].append(data[acc_num]['label'])
#         accept_data['tag'].append(data[acc_num]['tag'])
#         
#         rej_num = data['tag']==1
#         reject_data['label'].append(data[rej_num]['label'])
#         reject_data['tag'].append(data[rej_num]['tag'])
#     
#     #
#     accept_data['label'] = np.concatenate([x.values for x in accept_data['label']])
#     accept_data['tag'] = np.concatenate([x.values for x in accept_data['tag']])
#     reject_data['label'] = np.concatenate([x.values for x in reject_data['label']])
#     reject_data['tag'] = np.concatenate([x.values for x in reject_data['tag']])
#     
#     print("accepted sample number is {},default rate is {:.4f},dafualt number {},other number {} "\
#           .format(len(accept_data['label']),np.mean(accept_data['label']),\
#           np.sum(accept_data['label']),len(accept_data['label']) - np.sum(accept_data['label']) ) )
#     
#     reject_label = ~np.isnan(reject_data['label'])
#     print("rejected sample but have label {},default rate is {:.4f},dafualt number {},other number {} "\
#           .format( np.sum(reject_label),np.mean(reject_data['label'][reject_label]),\
#                   ))
#     
# =============================================================================
        
        