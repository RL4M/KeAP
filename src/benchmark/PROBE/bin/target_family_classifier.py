# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:32:26 2020

@author: Muammer
"""

import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd 
from numpy import save
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import math


representation_name = ""
representation_path = ""
dataset = "nc"
detailed_output = False

def convert_dataframe_to_multi_col(representation_dataframe):
    entry = pd.DataFrame(representation_dataframe['Entry'])
    vector = pd.DataFrame(list(representation_dataframe['Vector']))
    multi_col_representation_vector = pd.merge(left=entry,right=vector,left_index=True, right_index=True)
    return multi_col_representation_vector

def class_based_scores(c_report, c_matrix):
    c_report = pd.DataFrame(c_report).transpose()
    #print(c_report)
    c_report = c_report.drop(['precision', 'recall'], axis=1)
    c_report = c_report.drop(labels=['accuracy', 'macro avg', 'weighted avg'], axis=0)
    cm = c_matrix.astype('float') / c_matrix.sum(axis=1)[:, np.newaxis]
    #print(c_report)
    accuracy = cm.diagonal()
    
    #print(accuracy)
    #if len(accuracy) == 6:
     #   accuracy = np.delete(accuracy, 5)

    accuracy = pd.Series(accuracy, index=c_report.index)
    c_report['accuracy'] = accuracy
    
    total = c_report['support'].sum()
    #print(total)
    num_classes = np.shape(c_matrix)[0]
    mcc = np.zeros(shape=(num_classes,), dtype='float32')
    weights = np.sum(c_matrix, axis=0)/np.sum(c_matrix)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for j in range(num_classes):
        tp = np.sum(c_matrix[j, j])
        fp = np.sum(c_matrix[j, np.concatenate((np.arange(0, j), np.arange(j+1, num_classes)))])
        fn = np.sum(c_matrix[np.concatenate((np.arange(0, j), np.arange(j+1, num_classes))), j])
        tn = int(total - tp - fp - fn)
        total_tp = total_tp + tp
        total_fp = total_fp + fp
        total_fn = total_fn + fn
        total_tn = total_tn + tn
        #print(tp,fp,fn,tn)
        mcc[j] = ((tp*tn)-(fp*fn))/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    #print(mcc)
    #if len(mcc) == 6:
     #   mcc = np.delete(mcc, 5)

    mcc = pd.Series(mcc, index=c_report.index)
    c_report['mcc'] = mcc
    #c_report.to_excel('../results/resultss_class_based_'+dataset+'.xlsx')
    #print(c_report)
    return c_report, total_tp, total_fp, total_fn, total_tn



def score_protein_rep(dataset):
#def score_protein_rep(pkl_data_path):

    vecsize = 0
    #protein_list = pd.read_csv('../data/auxilary_input/entry_class.csv')
    protein_list = pd.read_csv('../data/preprocess/entry_class_nn.csv')
    dataframe = pd.read_csv(representation_path)
    #dataframe = convert_dataframe_to_multi_col(dataframe)    
    #dataframe = pd.read_pickle(pkl_data_path)
    vecsize = dataframe.shape[1]-1    
    x = np.empty([0, vecsize])
    xemp = np.zeros((1, vecsize), dtype=float)
    y = []
    ne = []

    print("\n\nPreprocessing data for drug-target protein family prediction...\n ")
    for index, row in tqdm(protein_list.iterrows(), total=len(protein_list)):
        pdrow = dataframe.loc[dataframe['Entry'] == row['Entry']]
        if len(pdrow) != 0:
            a = pdrow.loc[ : , pdrow.columns != 'Entry']
            a = np.array(a)
            a.shape = (1,vecsize)
            x = np.append(x, a, axis=0)
            y.append(row['Class'])
        else:
            ne.append(index)
            x = np.append(x, xemp, axis=0,)
            y.append(0.0)
            #print(index)

    x = x.astype(np.float64)
    y = np.array(y)
    y = y.astype(np.float64)
    #print(len(y))
    scoring = ['precision_weighted', 'recall_weighted', 'f1_weighted', 'accuracy']
    target_names = ['Enzyme', 'Membrane receptor', 'Transcription factor', 'Ion channel', 'Other']
    labels = [1.0, 11.0, 12.0, 1005.0, 2000.0]
    
    f1 = []
    accuracy = []
    mcc = []
    f1_perclass = []
    ac_perclass = []
    mcc_perclass = []
    sup_perclass = []
    report_list = []
    train_index = pd.read_csv('../data/preprocess/indexes/'+dataset+'_trainindex.csv')
    test_index = pd.read_csv('../data/preprocess/indexes/testindex_family.csv')
    train_index = train_index.dropna(axis=1) 
    test_index = test_index.dropna(axis=1)
    #print(train_index)
    #for index in ne:
       

    conf = pd.DataFrame()

    print('Producing protein family predictions...\n')
    for i in tqdm(range(10)): 
        clf = linear_model.SGDClassifier(class_weight="balanced", loss="log", penalty="elasticnet", max_iter=1000, tol=1e-3,random_state=i,n_jobs=-1)
        clf2 = OneVsRestClassifier(clf,n_jobs=-1)
        #print(test_index)
        train_indexx = train_index.iloc[i].astype(int)
        test_indexx = test_index.iloc[i].astype(int)
        #print(train_indexx)
        #train_indexx.drop(labels=ne)
        #print(type(train_indexx))
        for index in ne:
            
            train_indexx = train_indexx[train_indexx!=index]
            test_indexx = test_indexx[test_indexx!=index]
            


        train_X, test_X = x[train_indexx], x[test_indexx]
        train_y, test_y = y[train_indexx], y[test_indexx]

        clf2.fit(train_X, train_y)    
            
        #print(train_X)
        y_pred = clf2.predict(test_X)
           
        #y_pred = cross_val_predict(clf2, x, y, cv=10, n_jobs=-1)
        #mcc.append(matthews_corrcoef(test_y, y_pred, sample_weight = test_y))
        f1_ = f1_score(test_y, y_pred, average='weighted')
        f1.append(f1_)
        ac = accuracy_score(test_y, y_pred)
        accuracy.append(ac)
        c_report = classification_report(test_y, y_pred, target_names=target_names, output_dict=True)
        c_matrix = confusion_matrix(test_y, y_pred, labels=labels)

        conf = conf.append(pd.DataFrame(c_matrix, columns=['Enzymes', 'Membrane receptor', 'Transcription factor', 'Ion channel', 'Other']), ignore_index=True) 
        class_report, tp, fp, fn, tn = class_based_scores(c_report, c_matrix)
        
        #print(total_tp)
        mcc.append(((tp*tn)-(fp*fn))/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))

        
        f1_perclass.append(class_report['f1-score'])
        ac_perclass.append(class_report['accuracy'])
        mcc_perclass.append(class_report['mcc'])
        sup_perclass.append(class_report['support'])
        report_list.append(class_report)
    
    if detailed_output:
        conf.to_csv('../results/Drug_target_protein_family_classification_confusion_'+dataset+'_'+representation_name+'.csv', index=None)

    f1_perclass = pd.concat(f1_perclass, axis=1)
    ac_perclass = pd.concat(ac_perclass, axis=1)
    mcc_perclass = pd.concat(mcc_perclass, axis=1)
    sup_perclass = pd.concat(sup_perclass, axis=1)
    
    report_list = pd.concat(report_list, axis=1)
    report_list.to_csv('../results/Drug_target_protein_family_classification_class_based_results_'+dataset+'_'+representation_name+'.csv')
    
    report = pd.DataFrame()    
    f1mean = np.mean(f1, axis=0)
    #print(f1mean)
    f1mean = f1mean.round(decimals=5)
    f1std = np.std(f1).round(decimals=5)
    acmean = np.mean(accuracy, axis=0).round(decimals=5)
    acstd = np.std(accuracy).round(decimals=5)
    mccmean = np.mean(mcc, axis=0).round(decimals=5)
    mccstd = np.std(mcc).round(decimals=5)
    labels = ['Average Score', 'Standard Deviation']
    report['Protein Family'] = labels
    report['F1_score'] = [f1mean, f1std]
    report['Accuracy'] = [acmean, acstd]
    report['MCC'] = [mccmean, mccstd]

    report.to_csv('../results/Drug_target_protein_family_classification_mean_results_'+dataset+'_'+representation_name+'.csv',index=False)
    #report.to_csv('scores_general.csv')
    #print(report)   
    if detailed_output:
        save('../results/Drug_target_protein_family_classification_f1_'+dataset+'_'+representation_name+'.npy', f1)
        save('../results/Drug_target_protein_family_classification_accuracy_'+dataset+'_'+representation_name+'.npy', accuracy)
        save('../results/Drug_target_protein_family_classification_mcc_'+dataset+'_'+representation_name+'.npy', mcc) 
        save('../results/Drug_target_protein_family_classification_class_based_f1_'+dataset+'_'+representation_name+'.npy', f1_perclass)
        save('../results/Drug_target_protein_family_classification_class_based_accuracy_'+dataset+'_'+representation_name+'.npy', ac_perclass)
        save('../results/Drug_target_protein_family_classification_class_based_mcc_'+dataset+'_'+representation_name+'.npy', mcc_perclass) 
        save('../results/Drug_target_protein_family_classification_class_based_support_'+dataset+'_'+representation_name+'.npy', sup_perclass) 

