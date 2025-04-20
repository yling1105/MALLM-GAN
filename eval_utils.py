import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from typing import List, Tuple, Dict, Callable
from sklearn.neighbors import NearestNeighbors
from cython_metric import mixed_distance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, r2_score
from sklearn.preprocessing import LabelBinarizer
import json
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
import os
from sklearn.preprocessing import LabelEncoder
from flaml import AutoML


def _prepare_data_for_privacy_metrics(
    tgt_data: DataFrame,
    syn_data: DataFrame,
    meta_data: Dict,
    smoothing_factor: float,
) -> Tuple[DataFrame, DataFrame]:
    """
    Data preparation for privacy metrics

    For categorical, ordinal encoding based on joint set of target data and synthetic data.
    For numeric encoding, missing value are imputed with mean and standardized

    :param tgt_data: pandas dataframe
    :param syn_data: pandas dataframe
    :param column_dictionary: column to type mapping
    :param smoothing_factor: smoothing factor


    :returns: privacy ready target + synthetic  (pamdas DataFrame)
    """

    tgt_data_p = tgt_data.copy(deep=True)
    syn_data_p = syn_data.copy(deep=True)

    for column_name, column_type in meta_data['columns'].items():
        if column_type['sdtype'] == "categorical":

            tgt_data_p[column_name] = tgt_data_p[column_name].cat.codes
            syn_data_p[column_name] = syn_data_p[column_name].cat.codes

        else:
            # fill na data with mean
            tgt_data_p[column_name] = tgt_data_p[column_name].fillna(
                tgt_data_p[column_name].dropna().mean()
            )
            syn_data_p[column_name] = syn_data_p[column_name].fillna(
                syn_data_p[column_name].dropna().mean()
            )

            # standardize
            tgt_data_p[column_name] = (
                tgt_data_p[column_name] - tgt_data_p[column_name].mean()
            ) / np.max([tgt_data_p[column_name].std(), smoothing_factor])
            syn_data_p[column_name] = (
                syn_data_p[column_name] - syn_data_p[column_name].mean()
            ) / np.max([syn_data_p[column_name].std(), smoothing_factor])
    return tgt_data_p, syn_data_p

def _get_nn_model(train: DataFrame, cat_slice) -> Tuple[np.ndarray]:
    nearest_neighbor_model = NearestNeighbors(
        metric=lambda x, y: mixed_distance(x, y, cat_slice),
        algorithm="ball_tree",
        n_jobs=None,
    )

    nearest_neighbor_model.fit(train)
    return nearest_neighbor_model

def _calculate_dcr(tgt_data0, syn_data0, meta_data: dict):
    category_cols = [col for col, val in meta_data['columns'].items() if val['sdtype'] == 'categorical']
    numerical_cols = [col for col, val in meta_data['columns'].items() if val['sdtype'] != 'categorical']
    tgt_data = tgt_data0.copy()
    syn_data = syn_data0.copy()
    # Encode categorical data
    label_encoders = {}
    df_comb = pd.concat([tgt_data, syn_data])
    for col in category_cols:
        le = LabelEncoder()
        try:
            le.fit(df_comb[col])
        except:
            print(col)
        tgt_data[col] = le.transform(tgt_data[col])
        syn_data[col] = le.transform(syn_data[col])
        label_encoders[col] = le

    # Reorder the data to have categorical columns first
    tgt_data = pd.concat([tgt_data[category_cols], tgt_data[numerical_cols]], axis=1)
    syn_data = pd.concat([syn_data[category_cols], syn_data[numerical_cols]], axis=1)

    # Initialize the NearestNeighbors model
    cat_slice = len(category_cols)
    nn_model = _get_nn_model(tgt_data, cat_slice)
    syn_query_neighbors = nn_model.kneighbors(syn_data, n_neighbors=2)
    dcr = syn_query_neighbors[0][:, 0]
    df_privacy = pd.DataFrame({"DCR": dcr})

    return df_privacy


def compare_dcr(tgt_data, syn_data, meta_data):
    smoothing_factor = 1e-08
    for col in tgt_data.select_dtypes(include=['object', 'boolean']).columns:
        tgt_data[col] = tgt_data[col].astype('category')

    for col in syn_data.select_dtypes(include=['object']).columns:
        syn_data[col] = tgt_data[col].astype('category')
    tgt_data_p, syn_data_p = _prepare_data_for_privacy_metrics(tgt_data, syn_data, meta_data, smoothing_factor)

    dcr = _calculate_dcr(tgt_data_p, syn_data_p, meta_data)
    return dcr

def compare_MLE(X_test, y_test, X_train, y_train, task, seed = 1234):
    '''
    Train machine learning models on the synthetic dataset and then test it on the real data. 
    '''
    performance = {}
    classification_model = {
        "Logistic Regression": LogisticRegression(random_state=seed, max_iter=3000),
        "Random Forest": RandomForestClassifier(max_depth=3, n_jobs=10, random_state=seed),
        "SVC": SVC(random_state=seed),
        "XGB": XGBClassifier(
                    objective='binary:logistic',  # for binary classification, change for multiclass
                    n_estimators=100,             # number of boosting rounds
                    learning_rate=0.1,            # step size shrinkage
                    max_depth=3,                  # maximum depth of a tree
                    eval_metric='logloss'         # evaluation metric
                )
    }

    regression_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=3, random_state=seed),
        "XGB": XGBRegressor(
                objective='reg:squarederror',  # regression with squared loss
                n_estimators=100,              # number of boosting rounds
                learning_rate=0.1,             # step size shrinkage
                max_depth=3,                   # maximum depth of a tree
        )
    }

    if task == 'classification':
        for name, model in classification_model.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            f1 = f1_score(y_test, predictions, average="weighted")
            accu = accuracy_score(y_test, predictions)
            performance[name] = {}
            performance[name]['f1 score'] = f1
            performance[name]['accuracy'] = accu
    if task == 'regression':
        for name, model in regression_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_linear = r2_score(y_test, y_pred)
            performance[name] = {}
            performance[name]['R2'] = r2_linear
    
    return performance

def data_profiling(data, cate_cols, bool_cols, cols):
    schema = 'Schema:\n'
    for col in cols:
        if col in cate_cols:
            schema += (col + ' (categorical), ')
        elif col in bool_cols:
            schema += (col + ' (boolean), ')
        else:
            schema += (col + ' (numerical),')
    schema += '\n\n'
    return schema

def cate_info(data, cate_cols):
    schema = ""
    schema += 'Categorical variable and their available categories are:\n'
    for cate_col in cate_cols:
        cate_set = set(data[cate_col])
        schema += cate_col + f': {cate_set}, '
    schema += '\n'
    return schema

def col_check(df, cols):
    if set(cols) - set(df.columns):
        diff = list(set(cols) - set(df.columns))
        for col in diff:
            df[col] = [0] * len(df)
    if set(df.columns) - set(cols):
        dropped_cols = list(set(df.columns) - set(cols))
        df.drop(columns=dropped_cols, inplace=True)
    return df

def transform_label(df, y_col):
    lb = LabelBinarizer()
    y = lb.fit_transform(df[y_col]).ravel()
    return lb, y

def mle_summary(json_file):
    with open(json_file, 'r') as f:
        res = json.load(f)

    eval_sum_counts = {}
    for num, results in res.items():
        eval_sum_counts[num] = {}
        for seed, results0 in results.items():
            for model, results00 in results0.items():
                if model not in eval_sum_counts[num]:
                    eval_sum_counts[num][model] = {}
                for eval_model, results000 in results00.items():
                    if eval_model not in eval_sum_counts[num][model]:
                        eval_sum_counts[num][model][eval_model] = {}
                    for eval_metrics, results0000 in results000.items():
                        if eval_metrics not in eval_sum_counts[num][model][eval_model]:
                            eval_sum_counts[num][model][eval_model][eval_metrics] = []
                        eval_sum_counts[num][model][eval_model][eval_metrics].append(results0000)
    res_average = {}
    for num, results in eval_sum_counts.items():
        res_average[num] = {}
        for model, results00 in results.items():
            res_average[num][model] = {}
            for eval_model, results000 in results00.items():
                res_average[num][model][eval_model] = {}
                for eval_metrics, results0000 in results000.items():
                    res_average[num][model][eval_model][eval_metrics] = {}
                    res_average[num][model][eval_model][eval_metrics]['mean'] = np.mean(results0000)
                    res_average[num][model][eval_model][eval_metrics]['std'] = np.std(results0000)
    return res_average

def get_progression(num_samples, epochs, model_dict, report_cols, metadata):
    overall_score = {}
    for i in range(4):
        num = num_samples[i]
        overall_score[str(num)] = {}
        for j in range(5):
            index = str(i) + '-' + str(j)
            mallm_temp = model_dict[index]
            real_data_temp = mallm_temp.real_data
        
            for e in range(epochs[i]):
                ii = 0
                df_res_temp = []
                if str(e) not in overall_score[str(num)]:
                    overall_score[str(num)][str(e)] = []
                while ii < len(mallm_temp.real_data):
                    df_idx = str(e)+'-'+ str(ii)
                    res_temp = mallm_temp.process_response(mallm_temp.res[df_idx])
                    df_res_temp.append(res_temp)
                    print(len(df_res_temp))
                    ii += 40
                df_res = pd.concat(df_res_temp)
                df_res.reset_index(inplace=True, drop=True)
            report = QualityReport()
            report.generate(real_data_temp[report_cols], df_res[report_cols], metadata)
            overall_score[str(num)][str(e)].append(report._overall_score)
    return overall_score

def shape_summary(path, df_train, meta_data, cols):
    over_all_score = []
    for file_name in os.listdir(path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(path, file_name)
            df_temp = pd.read_csv(path + file_name)
            df_temp.reset_index(inplace=True, drop=True)
            report = QualityReport()
            report.generate(df_train[cols], df_temp[cols], meta_data)
            over_all_score.append(report._overall_score)
    return over_all_score

def shape_summary_from_json(path):
    summary_shape = {}
    with open(path, 'r') as f:
        res = json.load(f)
    
    for num, res0 in res.items():
        summary_shape[num] = {}
        for model, res00 in res0.items():
            summary_shape[num][model] = np.mean(res00)
    return summary_shape

def plot_mle_summary(o_path, m_path):
    '''
    o_path: other model's results
    m_path: mallm_gan model's results
    '''
    with open(o_path, 'r') as f:
        o_res = json.load(f)
    
    with open(m_path, 'r') as f:
        m_res = json.load(f)

    records = []
    for sample_size, seeds in o_res.items():
        for seed, models in seeds.items():
            for model_name, ml_models in models.items():
                for ml_model, metrics in ml_models.items():
                    record = {
                        'Sample Size': sample_size,
                        'Seed': seed,
                        'Model': model_name,
                        'ML Model': ml_model,
                        'Accuracy': metrics['accuracy']
                    }
                    records.append(record)
    for sample_size, seeds in m_res.items():
        for seed, models in seed.items():
            for model_name, ml_models in models.items():
                for ml_model, metrics in ml_models.items():
                    record = {
                        'Sample Size': sample_size,
                        'Seed': seed,
                        'Model': model_name,
                        'ML Model': ml_model,
                        'Accuracy': metrics['accuracy']
                    }
                    records.append(record)
    return records

def plot_shape_summary(path):
    data = ['Adult', 'Insurance', 'ATACH', 'ERICH']
    num_samples = [100, 200, 400, 800]
    models = ['CTGAN', 'TVAE', 'BeGreaT', 'MALLM-GAN']
    records = []
    for df in data:
        oracle = pd.read_csv(f'original_data/{df}.csv')
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        meta_data = metadata.to_dict()
        for num in num_samples:
            for model_nm in models:
                df_lst = []
                for seed in range(5):
                    df_temp = pd.read_csv(f'df_{seed}.csv')
                    report = QualityReport()
                    report.generate(oracle, df_temp, meta_data)
                    record = {
                        'Dataset': df,
                        'Sample size': num,
                        'Seed': seed,
                        'Model': model_nm,
                        'Score': report._over_all_score
                    }
    
    return pd.DataFrame(records)



def compare_MLE_autoML(X_test, y_test, X_train, y_train, task, seed=1234, time_budget=60):
    '''
    Use AutoML to find the best model on synthetic training data and evaluate on real test data.
    '''
    automl = AutoML()
    performance = {}

    if task == 'classification':
        automl_settings = {
            "time_budget": time_budget,  # seconds
            "metric": 'f1',
            "task": 'classification',
            "log_file_name": "automl_classification.log",
            "seed": seed
        }

    elif task == 'regression':
        automl_settings = {
            "time_budget": time_budget,
            "metric": 'r2',
            "task": 'regression',
            "log_file_name": "automl_regression.log",
            "seed": seed
        }

    # Run AutoML search
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

    # Predict and score
    y_pred = automl.predict(X_test)

    if task == 'classification':
        performance['AutoML'] = {
            'f1 score': f1_score(y_test, y_pred, average='weighted'),
            'accuracy': accuracy_score(y_test, y_pred),
            'best_model': str(automl.model)
        }
    else:
        performance['AutoML'] = {
            'R2': r2_score(y_test, y_pred),
            'best_model': str(automl.model)
        }

    return performance

