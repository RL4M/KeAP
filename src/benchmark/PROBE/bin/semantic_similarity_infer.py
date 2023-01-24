#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import gzip
import itertools
import multiprocessing
import csv
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity as cosine
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm, tqdm_notebook
from multiprocessing import Manager, Pool
from scipy.spatial.distance import cdist
from numpy.linalg import norm
from scipy.stats import spearmanr, pearsonr
from functools import partial

manager = Manager()
similarity_list = manager.list()
proteinListNew = manager.list()
 
representation_dataframe = ""
protein_names =  ""
# define similarity_list and proteinList as global variables
representation_name = ""
similarity_tasks = ""
detailed_output = False

def parallelSimilarity(paramList):
    protein_embedding_dataframe = representation_dataframe
    i = paramList[0]
    j = paramList[1] 
    aspect = paramList[2]
    if j>i:
        protein1 = proteinListNew[i]
        protein2 = proteinListNew[j]
        if protein1 in protein_names and protein2 in protein_names:
            prot1vec = np.asarray(protein_embedding_dataframe.query("Entry == @protein1")['Vector'].item())
            prot2vec = np.asarray(protein_embedding_dataframe.query("Entry == @protein2")['Vector'].item())
            #cosine will return in shape of input vectors first dimension
            cos = cosine(prot1vec.reshape(1,-1),prot2vec.reshape(1,-1)).item()
            manhattanDist = cdist(prot1vec.reshape(1,-1), prot2vec.reshape(1,-1), 'cityblock')
            manhattanDistNorm = manhattanDist/(norm(prot1vec,1) + norm(prot2vec,1))
            manhattanSim = 1-manhattanDistNorm.item()
            if (norm(prot1vec,1)==0 and norm(prot2vec,1) == 0):
                manhattanSim = 1.0
                #print((protein1,protein2))
                #print(manhattanDist)
                #print(norm(prot1vec,1))
                #print(norm(prot2vec,1))
            euclideanDist = cdist(prot1vec.reshape(1,-1), prot2vec.reshape(1,-1), 'euclidean')
            euclideanDistNorm = euclideanDist/(norm(prot1vec,2) + norm(prot2vec,2))
            euclidianSim =  1-euclideanDistNorm.item()
            if (norm(prot1vec,1)==0 and norm(prot2vec,1) == 0):
                euclidianSim = 1.0
            real = paramList[3]
            # To ensure real and calculated values appended to same postion they saved similtanously and then decoupled
            similarity_list.append((real,cos,manhattanSim ,euclidianSim))
    return similarity_list

def calculateCorrelationforOntology(aspect,matrix_type):
    print("\n\nSemantic similarity correlation calculation for aspect: " + aspect + " using matrix/dataset: " + matrix_type + " ...\n")
    #Clear lists before each aspect
    similarity_list[:] = []
    proteinListNew[:] = []
    
    similarityMatrixNameDict = {}
    similarityMatrixNameDict["All"] = "../data/preprocess/human_"+aspect+"_proteinSimilarityMatrix.csv" 
    similarityMatrixNameDict["500"] = "../data/preprocess/human_"+aspect+"_proteinSimilarityMatrix_for_highest_annotated_500_proteins.csv"
    similarityMatrixNameDict["Sparse"] = "../data/preprocess/human_"+aspect+"_proteinSimilarityMatrix_for_highest_annotated_500_proteins.csv" 
    similarityMatrixNameDict["200"] = "../data/preprocess/human_"+aspect+"_proteinSimilarityMatrix_for_highest_annotated_200_proteins.csv"

    similarityMatrixFileName = similarityMatrixNameDict[matrix_type]

    human_proteinSimilarityMatrix = pd.read_csv(similarityMatrixFileName)
    human_proteinSimilarityMatrix.set_index(human_proteinSimilarityMatrix.columns, inplace = True)
    proteinList = human_proteinSimilarityMatrix.columns

    #proteinListNew is referanced using Manager
    for prot in proteinList:
        proteinListNew.append(prot)
    if matrix_type == "Sparse":
        #sparsified_similarities = np.load("SparsifiedSimilarites_for_highest_500.npy")
        sparsified_similarity_coordinates = np.load("../data/auxilary_input/SparsifiedSimilarityCoordinates_"+aspect+"_for_highest_500.npy")
        protParamList = sparsified_similarity_coordinates
    else:     
        i = range(len(proteinList))
        j = range(len(proteinList))
        protParamList = list(itertools.product(i,j))
    protParamListNew = []
    # Prepare parameters for parallel processing these parameters will be 
    # used concurrently by different processes
    for tup in tqdm(protParamList):
        i = tup[0]
        j = tup[1]

        if matrix_type == "Sparse":
            protein1 = proteinListNew[i]
            protein2 = proteinListNew[j]
            real = human_proteinSimilarityMatrix.loc[protein1,protein2]
            tupNew = (tup[0],tup[1],aspect,real)
            protParamListNew.append(tupNew)
        else:
            if j > i:
                protein1 = proteinListNew[i]
                protein2 = proteinListNew[j]
                real = human_proteinSimilarityMatrix.loc[protein1,protein2]
                tupNew = (tup[0],tup[1],aspect,real)
                protParamListNew.append(tupNew)

    total_task_num=len(protParamListNew)
    pool = Pool()
    similarity_listRet = []
    #parallelSimilarityPartial = partial(parallelSimilarity,protein_embedding_type)
    for similarity_listRet in tqdm(pool.imap_unordered(parallelSimilarity,protParamListNew), total=total_task_num , position=0, leave=True ):
        pass
        #time.sleep(0.1)
    pool.close()
    pool.join()

    real_distance_list = [value[0] for value in similarity_listRet]
    cosine_distance_list = [value[1] for value in similarity_listRet]
    manhattan_distance_list = [value[2] for value in similarity_listRet]
    euclidian_distance_list = [value[3] for value in similarity_listRet]

    distance_lists = [real_distance_list,cosine_distance_list,manhattan_distance_list,euclidian_distance_list]
    if detailed_output:
        report_detailed_distance_scores(representation_name,matrix_type,aspect,distance_lists)
    
    cosineCorr = spearmanr(real_distance_list, cosine_distance_list)
    manhattanCorr = spearmanr(real_distance_list, manhattan_distance_list)
    euclidianCorr = spearmanr(real_distance_list, euclidian_distance_list)   

    #print("Cosine Correlation for "+aspect+" is " + str(cosineCorr))
    #print("Manhattan Correlation for "+aspect+" is " + str(manhattanCorr))
    #print("Euclidian Correlation for "+aspect+" is " + str(euclidianCorr))

    return (cosineCorr,manhattanCorr,euclidianCorr)

def report_detailed_distance_scores(representation_name,similarity_matrix_type,aspect,distance_lists):
    saveFileName = "../results/Semantic_sim_inference_detailed_distance_scores"+aspect+"_"+similarity_matrix_type+"_"+representation_name+".pkl"
    with open(saveFileName, "wb") as f:
            pickle.dump(distance_lists, f)

def calculate_all_correlations():
    for similarity_matrix_type in similarity_tasks:
        saveFileName = "../results/Semantic_sim_inference_"+similarity_matrix_type+"_"+representation_name+".csv"
        buffer = "Semantic Aspect,CosineSim_Correlation,CosineSim_Correlation p-value, ManhattanSim_Correlation,ManhattanSim_Correlation p-value, EuclidianSim_Correlation,EuclidianSim_Correlation p-value \n"
        f = open(saveFileName,'w')
        f.write(buffer)
        for aspect in ["MF","BP","CC"]:
            corr =  calculateCorrelationforOntology(aspect,similarity_matrix_type) 
            buffer = "" + aspect + ","+ str(round(corr[0][0],5))+ ","+ str(round(corr[0][1],5))+ ","+ str(round(corr[1][0],5))\
            + ","+ str(round(corr[1][1],5))+ ","+ str(round(corr[2][0],5))+ ","+str(round(corr[2][1],5))+"\n" 
            f = open(saveFileName,'a')
            f.write(buffer)
            f.close()
