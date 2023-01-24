# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
import multiprocessing
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict, KFold
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss


aspect_type = ""
dataset_type = ""
representation_dataframe = ""
representation_name = ""
detailed_output = False

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def check_for_at_least_two_class_sample_exits(y):
    for column in y:
        column_sum = np.sum(y[column].array)
        if column_sum < 2:
           print('At least 2 positive samples are required for each class {0} class has {1} positive samples'.format(column,column_sum))
           return False
    return True

def create_valid_kfold_object_for_multilabel_splits(X,y,kf):
    check_for_at_least_two_class_sample_exits(y)
    sample_class_occurance = dict(zip(y.columns,np.zeros(len(y.columns))))
    for column in y:
        for fold_train_index,fold_test_index in kf.split(X,y):
            fold_col_sum = np.sum(y.iloc[fold_test_index,:][column].array)
            if fold_col_sum > 0:
                sample_class_occurance[column] += 1 

    for key in sample_class_occurance:
        value = sample_class_occurance[key]
        if value < 2:
            random_state = np.random.randint(1000)
            print("Random state is changed since at least two positive samples are required in different train/test folds.\
                    \nHowever, only one fold exits with positive samples for class {0}".format(key))
            print("Selected random state is {0}".format(random_state))
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            create_valid_kfold_object_for_multilabel_splits(X,y,kf)
        else:
            return kf

def MultiLabelSVC_cross_val_predict(representation_name, dataset, X, y, classifier):
    #dataset split, estimator, cv
    clf = classifier
    Xn = np.array(np.asarray(X.values.tolist()), dtype=float)
    kf_init = KFold(n_splits=5, shuffle=True, random_state=42)
    kf = create_valid_kfold_object_for_multilabel_splits(X,y,kf_init)
    y_pred = cross_val_predict(clf, Xn, y, cv=kf)

    if detailed_output:
        with open(r"../results/Ontology_based_function_prediction_{1}_{0}_model.pkl".format(representation_name,dataset.split(".")[0]),"wb") as file:
            pickle.dump(clf,file)
        
    acc_cv = []
    f1_mi_cv = []
    f1_ma_cv = []
    f1_we_cv = []
    pr_mi_cv = []
    pr_ma_cv = []
    pr_we_cv = []
    rc_mi_cv = []
    rc_ma_cv = []
    rc_we_cv = []
    hamm_cv = []
    for fold_train_index,fold_test_index in kf.split(X,y):
        acc = accuracy_score(y.iloc[fold_test_index,:],y_pred[fold_test_index])
        acc_cv.append(np.round(acc,decimals=5))
        f1_mi = f1_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="micro")
        f1_mi_cv.append(np.round(f1_mi,decimals=5))
        f1_ma = f1_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="macro")
        f1_ma_cv.append(np.round(f1_ma,decimals=5))
        f1_we = f1_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="weighted")
        f1_we_cv.append(np.round(f1_we,decimals=5))
        pr_mi = precision_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="micro")
        pr_mi_cv.append(np.round(pr_mi,decimals=5))
        pr_ma = precision_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="macro")
        pr_ma_cv.append(np.round(pr_ma,decimals=5))
        pr_we = precision_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="weighted")
        pr_we_cv.append(np.round(pr_we,decimals=5))
        rc_mi = recall_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="micro")
        rc_mi_cv.append(np.round(rc_mi,decimals=5))
        rc_ma = recall_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="macro")
        rc_ma_cv.append(np.round(rc_ma,decimals=5))
        rc_we = recall_score(y.iloc[fold_test_index,:],y_pred[fold_test_index],average="weighted")
        rc_we_cv.append(np.round(rc_we,decimals=5))
        hamm = hamming_loss(y.iloc[fold_test_index,:],y_pred[fold_test_index])
        hamm_cv.append(np.round(hamm,decimals=5))

    means = list(np.mean([acc_cv,f1_mi_cv,f1_ma_cv,f1_we_cv,pr_mi_cv,pr_ma_cv,pr_we_cv,rc_mi_cv,rc_ma_cv,rc_we_cv,hamm_cv], axis=1))
    means = [np.round(i,decimals=5) for i in means]

    stds = list(np.std([acc_cv,f1_mi_cv,f1_ma_cv,f1_we_cv,pr_mi_cv,pr_ma_cv,pr_we_cv,rc_mi_cv,rc_ma_cv,rc_we_cv,hamm_cv], axis=1))
    stds = [np.round(i,decimals=5) for i in stds]

    return ([representation_name+"_"+dataset,acc_cv,f1_mi_cv,f1_ma_cv,f1_we_cv,pr_mi_cv,pr_ma_cv,pr_we_cv,rc_mi_cv,rc_ma_cv,rc_we_cv,hamm_cv],\
            [representation_name+"_"+dataset]+means,\
            [representation_name+"_"+dataset]+stds,\
            y_pred)
   
def ProtDescModel():   
    #desc_file = pd.read_csv(r"protein_representations\final\{0}_dim{1}.tsv".format(representation_name,desc_dim),sep="\t")    
    datasets = os.listdir(r"../data/auxilary_input/GO_datasets") 
    if  dataset_type == "All_Data_Sets" and aspect_type == "All_Aspects":
        filtered_datasets = datasets
    elif dataset_type == "All_Data_Sets":
        filtered_datasets = [dataset for dataset in datasets if aspect_type in dataset]
    elif aspect_type == "All_Aspects":
        filtered_datasets = [dataset for dataset in datasets if dataset_type in dataset]
    else:
        filtered_datasets = [dataset for dataset in datasets if aspect_type in dataset and dataset_type in dataset]
    cv_results = []
    cv_mean_results = []
    cv_std_results = []

    for dt in tqdm(filtered_datasets,total=len(filtered_datasets)):
        print(r"Protein function prediction is started for the dataset: {0}".format(dt.split(".")[0]))
        dt_file = pd.read_csv(r"../data/auxilary_input/GO_datasets/{0}".format(dt),sep="\t")
        dt_merge = dt_file.merge(representation_dataframe,left_on="Protein_Id",right_on="Entry")

        dt_X = dt_merge['Vector']
        dt_y = dt_merge.iloc[:,1:-2]
        if check_for_at_least_two_class_sample_exits(dt_y) == False:
            print(r"No funtion will be predicted for the dataset: {0}".format(dt.split(".")[0]))
            continue
        #print("raw dt vs. dt_merge: {} - {}".format(len(dt_file),len(dt_merge)))
        #print("Calculating predictions for " +  dt.split(".")[0])
        #model = MultiLabelSVC_cross_val_predict(representation_name, dt.split(".")[0], dt_X, dt_y, classifier=BinaryRelevance(SVC(kernel="linear", random_state=42)))
        cpu_number  = multiprocessing.cpu_count()
        model = MultiLabelSVC_cross_val_predict(representation_name, dt.split(".")[0], dt_X, dt_y, classifier=BinaryRelevance(SGDClassifier(n_jobs=cpu_number, random_state=42)))
        cv_results.append(model[0])                
        cv_mean_results.append(model[1])
        cv_std_results.append(model[2])

        predictions = dt_merge.iloc[:,:6]
        predictions["predicted_values"] = list(model[3].toarray())
        if detailed_output:
            predictions.to_csv(r"../results/Ontology_based_function_prediction_{1}_{0}_predictions.tsv".format(representation_name,dt.split(".")[0]),sep="\t",index=None)

    return (cv_results, cv_mean_results,cv_std_results)             

#def pred_output(representation_name, desc_dim):
def pred_output():
    model = ProtDescModel()
    cv_result = model[0]
    df_cv_result = pd.DataFrame({"Model": pd.Series([], dtype='str') ,"Accuracy": pd.Series([], dtype='float'),"F1_Micro": pd.Series([], dtype='float'),\
            "F1_Macro": pd.Series([], dtype='float'),"F1_Weighted": pd.Series([], dtype='float'),"Precision_Micro": pd.Series([], dtype='float'),\
            "Precision_Macro": pd.Series([], dtype='float'),"Precision_Weighted": pd.Series([], dtype='float'),"Recall_Micro": pd.Series([], dtype='float'),\
            "Recall_Macro": pd.Series([], dtype='float'),"Recall_Weighted": pd.Series([], dtype='float'),"Hamming_Distance": pd.Series([], dtype='float')})
    for i in cv_result:
        df_cv_result.loc[len(df_cv_result)] = i
    if detailed_output:
        df_cv_result.to_csv(r"../results/Ontology_based_function_prediction_5cv_{0}.tsv".format(representation_name),sep="\t",index=None)

    cv_mean_result = model[1]
    df_cv_mean_result =  pd.DataFrame({"Model": pd.Series([], dtype='str') ,"Accuracy": pd.Series([], dtype='float'),"F1_Micro": pd.Series([], dtype='float'),\
            "F1_Macro": pd.Series([], dtype='float'),"F1_Weighted": pd.Series([], dtype='float'),"Precision_Micro": pd.Series([], dtype='float'),\
            "Precision_Macro": pd.Series([], dtype='float'),"Precision_Weighted": pd.Series([], dtype='float'),"Recall_Micro": pd.Series([], dtype='float'),\
            "Recall_Macro": pd.Series([], dtype='float'),"Recall_Weighted": pd.Series([], dtype='float'),"Hamming_Distance": pd.Series([], dtype='float')})

    
    #pd.DataFrame(columns=["Model","Accuracy","F1_Micro","F1_Macro","F1_Weighted","Precision_Micro","Precision_Macro","Precision_Weighted",\
    #                                     "Recall_Micro","Recall_Macro","Recall_Weighted","Hamming_Distance"])

    for j in cv_mean_result:
        df_cv_mean_result.loc[len(df_cv_mean_result)] = j
    df_cv_mean_result.to_csv(r"../results/Ontology_based_function_prediction_5cv_mean_{0}.tsv".format(representation_name),sep="\t",index=None)

#save std deviation of scores to file
    cv_std_result = model[2]
    df_cv_std_result =  pd.DataFrame({"Model": pd.Series([], dtype='str') ,"Accuracy": pd.Series([], dtype='float'),"F1_Micro": pd.Series([], dtype='float'),\
            "F1_Macro": pd.Series([], dtype='float'),"F1_Weighted": pd.Series([], dtype='float'),"Precision_Micro": pd.Series([], dtype='float'),\
            "Precision_Macro": pd.Series([], dtype='float'),"Precision_Weighted": pd.Series([], dtype='float'),"Recall_Micro": pd.Series([], dtype='float'),\
            "Recall_Macro": pd.Series([], dtype='float'),"Recall_Weighted": pd.Series([], dtype='float'),"Hamming_Distance": pd.Series([], dtype='float')})

    
    #pd.DataFrame(columns=["Model","Accuracy","F1_Micro","F1_Macro","F1_Weighted","Precision_Micro","Precision_Macro","Precision_Weighted",\
    #                                     "Recall_Micro","Recall_Macro","Recall_Weighted","Hamming_Distance"])

    for k in cv_std_result:
        df_cv_std_result.loc[len(df_cv_std_result)] = k
    df_cv_std_result.to_csv(r"../results/Ontology_based_function_prediction_5cv_std_{0}.tsv".format(representation_name),sep="\t",index=None)

print(datetime.now())      


# tcga = pred_output("tcga","50") 
# protvec = pred_output("protvec","100")  
# unirep = pred_output("unirep","5700")  
# gene2vec = pred_output("gene2vec","200")   
# learned_embed = pred_output("learned_embed","64") 
# mut2vec = pred_output("mut2vec","300")    
# seqvec = pred_output("seqvec","1024") 

#bepler = pred_output("bepler","100") 
# resnet_rescaled = pred_output("resnet-rescaled","256") 
# transformer_avg = pred_output("transformer","768") 
# transformer_pool = pred_output("transformer-pool","768") 

# apaac = pred_output("apaac","80") 
#ksep = pred_output("ksep","400") 

