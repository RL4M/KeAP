import tqdm
import multiprocessing
import pandas as pd
import numpy as np
import scipy.stats

from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

skempi_vectors_path = None
representation_name = None

def load_representation(multi_col_representation_vector_file_path):
    print("\nLoading representation vectors...\n")
    multi_col_representation_vector = pd.read_csv(multi_col_representation_vector_file_path)
    vals = multi_col_representation_vector.iloc[:,1:(len(multi_col_representation_vector.columns))]
    original_values_as_df = pd.DataFrame({'PDB_ID': pd.Series([], dtype='str'),'Vector': pd.Series([], dtype='object')})
    for index, row in tqdm.tqdm(vals.iterrows(), total = len(vals)):
        list_of_floats = [float(item) for item in list(row)]
        original_values_as_df.loc[index] = [multi_col_representation_vector.iloc[index]['PDB_ID']] + [list_of_floats]
    return original_values_as_df

def calc_train_error(X_train, y_train, model):
    '''returns in-sample error for already fit model.'''
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    mae = mean_absolute_error(y_train, predictions)
    corr = scipy.stats.pearsonr(y_train, predictions)
    return mse,mae,corr
    
def calc_validation_error(X_test, y_test, model):
    '''returns out-of-sample error for already fit model.'''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    corr = scipy.stats.pearsonr(y_test, predictions)
    return mse,mae,corr
    
def calc_metrics(X_train, y_train, X_test, y_test, model):
    '''fits model and returns the metrics for in-sample error and out-of-sample error'''
    model.fit(X_train, y_train)
    train_mse_error,train_mae_error,train_corr = calc_train_error(X_train, y_train, model)
    val_mse_error,val_mae_error,val_corr = calc_validation_error(X_test, y_test, model)
    return train_mse_error, val_mse_error, train_mae_error, val_mae_error,train_corr,val_corr

def report_results(
    train_mse_error_list,
    validation_mse_error_list,
    train_mae_error_list,
    validation_mae_error_list,
    train_corr_list,
    validation_corr_list,
    train_corr_pval_list,
    validation_corr_pval_list,
):
    result_df = pd.DataFrame(
        {
            "train_mse_error": round(np.mean(train_mse_error_list) * 100, 4),
            "train_mse_std": round(np.std(train_mse_error_list) * 100, 4),
            "val_mse_error": round(np.mean(validation_mse_error_list) * 100, 4),
            "val_mse_std": round(np.std(validation_mse_error_list) * 100, 4),
            "train_mae_error": round(np.mean(train_mae_error_list) * 100, 4),
            "train_mae_std": round(np.std(train_mae_error_list) * 100, 4),
            "val_mae_error": round(np.mean(validation_mae_error_list) * 100, 4),
            "val_mae_std": round(np.std(validation_mae_error_list) * 100, 4),
            "train_corr": round(np.mean(train_corr_list), 4),
            "train_corr_pval": round(np.mean(train_corr_pval_list), 4),
            "validation_corr": round(np.mean(validation_corr_list), 4),
            "validation_corr_pval": round(np.mean(validation_corr_pval_list), 4),
        },
        index=[0],
    )

    result_detail_df = pd.DataFrame(
        {
            "train_mse_errors": list(np.multiply(train_mse_error_list, 100)),
            "val_mse_errors": list(np.multiply(validation_mse_error_list, 100)),
            "train_mae_errors": list(np.multiply(train_mae_error_list, 100)),
            "val_mae_errors": list(np.multiply(validation_mae_error_list, 100)),
            "train_corrs": list(np.multiply(train_corr_list, 100)),
            "train_corr_pvals": list(np.multiply(train_corr_pval_list, 100)),
            "validation_corr": list(np.multiply(validation_corr_list, 100)),
            "validation_corr_pval": list(np.multiply(validation_corr_pval_list, 100)),
        },
        index=range(len(train_mse_error_list)),
    )
    return result_df, result_detail_df


def predictAffinityWithModel(regressor_model,multiplied_vectors_df):
    K = 10
    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    train_mse_error_list = []
    validation_mse_error_list = []
    train_mae_error_list = []
    validation_mae_error_list = []
    train_corr_list = []
    validation_corr_list = []
    train_corr_pval_list = []
    validation_corr_pval_list = []

    data = np.array(np.asarray(multiplied_vectors_df["Vector"].tolist()), dtype=float)
    ppi_affinity_filtered_df = ppi_affinity_df\
    [ppi_affinity_df['Protein1'].isin(multiplied_vectors_df['Protein1']) &\
     ppi_affinity_df['Protein2'].isin(multiplied_vectors_df['Protein2']) ]
    target = np.array(ppi_affinity_filtered_df["Affinity"])
    scaler = MinMaxScaler()
    scaler.fit(target.reshape(-1, 1))
    target = scaler.transform(target.reshape(-1, 1))[:, 0]
    for train_index, val_index in tqdm.tqdm(kf.split(data, target), total=K):

        # split data
        X_train, X_val = data[train_index], data[val_index]
        y_train, y_val = target[train_index], target[val_index]

        # instantiate model
        reg = regressor_model #linear_model.BayesianRidge()

        # calculate error_list
        (
            train_mse_error,
            val_mse_error,
            train_mae_error,
            val_mae_error,
            train_corr,
            val_corr,
        ) = calc_metrics(X_train, y_train, X_val, y_val, reg)

        # append to appropriate list
        train_mse_error_list.append(train_mse_error)
        validation_mse_error_list.append(val_mse_error)

        train_mae_error_list.append(train_mae_error)
        validation_mae_error_list.append(val_mae_error)

        train_corr_list.append(train_corr[0])
        validation_corr_list.append(val_corr[0])

        train_corr_pval_list.append(train_corr[1])
        validation_corr_pval_list.append(val_corr[1])

    return report_results(
        train_mse_error_list,
        validation_mse_error_list,
        train_mae_error_list,
        validation_mae_error_list,
        train_corr_list,
        validation_corr_list,
        train_corr_pval_list,
        validation_corr_pval_list,
    )

ppi_affinity_file = "../data/auxilary_input/skempi_pipr/SKEMPI_all_dg_avg.txt"
ppi_affinity_df = pd.read_csv(ppi_affinity_file,sep="\t",header=None)
ppi_affinity_df.columns = ['Protein1', 'Protein2', 'Affinity']

#Calculate vector element-wise multiplication as described in https://academic.oup.com/bioinformatics/article/35/14/i305/5529260

def calculate_vector_multiplications(skempi_vectors_df):
    multiplied_vectors = pd.DataFrame({'Protein1': pd.Series([], dtype='str'),\
                                       'Protein2': pd.Series([], dtype='str'),\
                                       'Vector': pd.Series([], dtype='object')}) 
    print("Element-wise vector multiplications are being calculated")
    rep_prot_list = list(skempi_vectors_df['PDB_ID'])
    for index,row in tqdm.tqdm(ppi_affinity_df.iterrows()):
        if row['Protein1'] in rep_prot_list and row['Protein2'] in rep_prot_list:
            vec1 = list(skempi_vectors_df[skempi_vectors_df['PDB_ID']\
                                                 == row['Protein1']]['Vector'])[0]
            vec2 = list(skempi_vectors_df[skempi_vectors_df['PDB_ID']\
                                                 == row['Protein2']]['Vector'])[0]
            multiplied_vec = np.multiply(vec1,vec2)

            multiplied_vectors = multiplied_vectors.\
                append({'Protein1':row['Protein1'], 'Protein2':row['Protein2'],\
                        'Vector':multiplied_vec},ignore_index = True)
    return multiplied_vectors

def predict_affinities_and_report_results():
    skempi_vectors_df = load_representation(skempi_vectors_path)
    multiplied_vectors_df = calculate_vector_multiplications(skempi_vectors_df)
    model = linear_model.BayesianRidge()
    result_df, result_detail_df = predictAffinityWithModel(model,multiplied_vectors_df)
    result_df.to_csv(r"../results/Affinity_prediction_skempiv1_{0}.csv".format(representation_name),index=False)
    result_detail_df.to_csv(r"../results/Affinity_prediction_skempiv1_{0}_detail.csv".format(representation_name),index=False)

