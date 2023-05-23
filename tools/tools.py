import os, shutil

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

import pandas as pd
import numpy as np

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.stats as stats

import seaborn as sns

from datetime import datetime

import re

from tqdm.notebook import tqdm

import dask.dataframe as dd

from typing import List, Dict

month_names = ['Gener','Febrer','Marc','Abril','Maig','Juny','Juliol','Agost','Setembre','Octubre','Novembre','Desembre']
months = range(1,13)
i2m = list(zip(months, month_names))

# study of skewness of the data population
def skewness(s:pd.Series) -> float:
    """
    Mesures of asymetry: 
    Negative deviation indicates that the destribution skews left. The skewness of a normal distrubtion is zero. And any symetric data must have skewness equal to zero.
    The alternative to this is by lookig into the relationship between the mean and the median.
    """
    
    assert not s.empty
    
    data_count = s.shape[0]
    assert data_count > 0
    
    data_mean = s.mean()
    data_std = s.std()
    
    result = 0
    for i in s:
        result += ((i - data_mean) * (i - data_mean) * (i - data_mean))
    result /= (data_count * data_std * data_std * data_std)
    
    return result

# Pearson median skewness coefficient
def pearson(df:pd.Series) -> float:
    """
    is an alternative to skewness coefficient
    """
    
    assert not s.empty
    
    data_count = s.shape[0]
    assert data_count > 0
    
    data_mean = s.mean()
    data_median = s.median()
    data_std = s.std()
    data_count = s.shape[0]
    
    result = 3*(data_mean - data_median)*data_std
    return result

def get_features_nans(df) -> Dict[str, float]:    
    result = None
    if type(df) == pd.DataFrame:
        result = (df.isna().sum()/df.shape[0])*100
    elif type(df) == dd.core.DataFrame:
        result = (df.isna().sum().compute()/df.shape[0].compute())*100
    else:
        raise Exception('Datatype not supported yet')
    return result[result > 0].to_dict()

def get_features_zero(df:pd.DataFrame) -> Dict[str, float]:
    result = None
    if type(df) == pd.DataFrame:
        result = (df.isin([0]).sum()/df.shape[0])*100
    elif type(df) == dd.core.DataFrame:
        result = (df.isin([0]).sum().compute()/df.shape[0].compute())*100
    else:
        raise Exception('Datatype not supported yet')
    return result[result > 0].to_dict()

def get_columns_nunique(df, cat_only:bool=False, num_only:bool=False) -> dict:
    
    assert (cat_only and num_only) is not True, 'can\'t be both true'
    
    columns = df.columns
    
    if cat_only:
        columns = df.select_dtypes(include=['object']).columns    
    
    if num_only:
        columns = df.select_dtypes(exclude=['object']).columns
    
    if type(df) == pd.DataFrame:
        return {column:df[column].nunique() for column in columns}
    elif type(df) == dd.core.DataFrame:
        return {column:df[column].nunique().compute() for column in columns}
    else:
        raise Exception('Datatype not supported yet')

def get_columns_unique(df, cat_only:bool=False, num_only:bool=False) -> dict:
    
    assert (cat_only and num_only) is not True, 'can\'t be both true'
    
    columns = df.columns
    
    if cat_only:
        columns = df.select_dtypes(include=['object']).columns    
    
    if num_only:
        columns = df.select_dtypes(exclude=['object']).columns
    
    if type(df) == pd.DataFrame:
        return {column:df[column].unique() for column in columns}
    elif type(df) == dd.core.DataFrame:
        return {column:df[column].unique().compute() for column in columns}
    else:
        raise Exception('Datatype not supported yet')

def cross_val_evaluation(model,X_train, y_train, model_name):
    scores = cross_val_score(model, X_train, y_train,cv=5) # scoring="neg_root_mean_squared_error"
    print("\n ",model_name)
    display_scores(scores)
    
def calcualte_scores(y, y_hat, show=True):
    ## Evaluate the model and plot it
    mdl_mse = mean_squared_error(y, y_hat)
    mdl_rmse = np.sqrt(mdl_mse)
    mdl_mae = mean_absolute_error(y, y_hat)
    mdl_r2score = r2_score(y, y_hat)
    
    # Best possible score is 1.0, lower values are worse.
    if show:
        print("----- EVALUATION ON VAL SET ------")
        print('MSE:', mdl_mse)
        print('RMSE', mdl_rmse)
        print('MAE:', mdl_mae)
        print('R^2: ', mdl_r2score) 
        print()
        plt.scatter(y, y_hat)
        plt.xlabel('y')
        plt.ylabel('y^')
        plt.show()
    return mdl_mse,mdl_rmse,mdl_mae,mdl_r2score

def show_column_counts(df:pd.DataFrame, column:str) -> None:
    assert column != ''
    assert df[column] is not None
    
    show_counts(df[column])

def show_counts(s:pd.Series) -> None:
    
    assert not s.empty
    
    data_count = s.shape[0]
    assert data_count > 0
    
    fig, axs = plt.subplots(3, 2, figsize=(20,10))
    axs[0][0].hist(s, label=f'{s.name} hist',bins=40)
    axs[0][0].set_xlabel('values')
    axs[0][0].set_ylabel('counts')
    axs[0][0].set_title('')

    axs[0][1].scatter(s.index, s.values, label=f'{s.name} scatter')
    axs[0][1].set_xlabel('index')
    axs[0][1].set_ylabel('values')
    axs[0][1].set_title('')
    
    axs[1][0].scatter(s.value_counts().index, s.value_counts().values,label=f'{s.name} counts')
    axs[1][0].set_xlabel('values')
    axs[1][0].set_ylabel('counts')
    axs[1][0].set_title('')
    
    axs[1][1].hist(s.value_counts(),label=f'{s.name} counts', bins=s.value_counts().shape[0])
    axs[1][1].set_xlabel('counts')
    axs[1][1].set_ylabel('values')
    axs[1][1].set_title('')
    
    
    axs[2][0].hist(s, density=True, histtype='step', cumulative=True,  linewidth=3.5, bins=30, color=sns.desaturate("indianred", .75))
    axs[2][0].set_xlabel('values')
    axs[2][0].set_ylabel('counts')
    axs[2][0].set_title('')
    
    axs[2][1].boxplot(s)
    axs[2][1].set_xlabel('counts')
    axs[2][1].set_ylabel('values')
    axs[2][1].set_title('')

    plt.tight_layout()
    plt.show()    
    
# We need to convert the timestamp column to datetime and merge the two datasets considering year, month, day and hour. (minutes and seconds will be merged and replaced with the mean)
from typing import List
def convert_timestamp(df:pd.DataFrame, columns:List[str], sort:bool=False, add:bool=False, unit:str='s', pattern:str=None) -> pd.DataFrame: 

    for column in columns: 
        if pattern:
            df[f'{column}_date'] = pd.to_datetime(df[column], format=pattern)
        else:
            df[f'{column}_date'] = pd.to_datetime(df[column], unit=unit)
        if add:
            df = add_time_columns(df, f'{column}_date')
            df.drop(f'{column}_date', axis=1, inplace=True)
            
    if sort:
        df = df.sort_values(columns, ascending=True).reset_index(drop=True)
        
    return df

def add_time_columns(df:pd.DataFrame, column:str):
    assert column != ''
    assert df[column] is not None
    
    df[f'year_{column}'] = df[column].dt.year
    df[f'month_{column}'] = df[column].dt.month
    df[f'week_{column}'] = df[column].dt.isocalendar().week
    df[f'dayofweek_{column}'] = df[column].dt.dayofweek
    df[f'dayofmonth_{column}'] = df[column].dt.day
    df[f'dayofyear_{column}'] = df[column].dt.dayofyear
    df[f'hour_{column}'] = df[column].dt.hour
    df[f'minutes_{column}'] = df[column].dt.minute
    
    return df
    
# This function works only for data of one station
def remove_duplicates(df:pd.DataFrame, column:str) -> pd.DataFrame:
    
    aux = df[column].value_counts()
    repeated_data = aux[aux > 1]

    for value in repeated_data.index:
        index = df[column] == value
        aux = df[index]  # taking only the ones with ttl bigger then 10
        
        candidates = aux.loc[aux['ttl'] > 10, :]
        candidates = candidates if candidates.shape[0] > 1 else aux

        cat_cols = df.select_dtypes(include=['object']).columns
        num_cols = df.select_dtypes(exclude=['object']).columns

        aux = candidates.mean() #.round().astype(np.int)
        
        for cat_col in cat_cols:
            value_counts_sorted = candidates.dropna()[cat_col].value_counts()
            aux[cat_col] = value_counts_sorted.index[0] if not value_counts_sorted.empty else np.nan
        
        assert df.shape[1] == aux.shape[0]
        
        df.drop(df[index].index, inplace=True)
        df = df.append(aux, ignore_index=True)
        
    # reorder the list
    df = df.sort_values(column, ascending=True).reset_index(drop=True)
    
    return df

def remove_duplicates_all(df:pd.DataFrame, column:str) -> pd.DataFrame:
    assert column != ''
    assert df[column] is not None
    
    result = {}
    
    for station_id in tqdm(df.station_id.unique().tolist()):
        df_s = df[df.station_id == station_id]
        df_s = remove_duplicates(df_s.copy(), column)
        result[station_id] = df_s.copy()
    
    # concat the result values
    df_ = pd.concat(list(result.values()), axis=0)
    
    return df_

def timestamp_multipleof(
    devide_by:int, 
    column:str,
    df:pd.DataFrame, 
    new_column:str, 
    year_column:str,
    month_column:str,
    day_column:str,
    hour_column:str,
    minutes_column:str
) -> pd.DataFrame:    
    
    assert column != ''
    assert df[column] is not None
    
    # convert time to multiples of 3
    df.loc[:,[column]] = (df[column]/devide_by).apply(np.floor)*devide_by

    # create mew column of last reported and last updated 
    df[new_column] = df.apply(
        lambda x: 
        datetime(
            year=int(x[year_column]), 
            month=int(x[month_column]), 
            day=int(x[day_column]), 
            hour=int(x[hour_column]), 
            minute=int(x[minutes_column]), 
        ),
        axis=1
    )
    
    # recommended method to convert datetime to integer timestamp 
    dates = df[new_column]
    # calculate unix datetime
    df[new_column] = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    return df

def print_duplicates(df:pd.DataFrame, columns:list):
    # check if conversion was done correctly
    return df.groupby(columns).nunique().max()

def correct_columns(df:pd.DataFrame, prim_column:str, column:str, drop:bool=True, correct_column:pd.DataFrame=pd.DataFrame(), take:str='max'):
    #print(df.shape)
    #print(column)
    
    if correct_column.empty:
        aux = pd.DataFrame()
        aux['unique'] = df.dropna()[[prim_column,column]].value_counts().reset_index().groupby([prim_column])[column].unique()
        if take == 'first':
            aux.loc[:, column] = [l[0] if len(l) > 0 else np.nan for l in aux.loc[:, 'unique']]
        elif take == 'max':
            aux.loc[:, column] = [l.max() if len(l) > 0 else np.nan for l in aux.loc[:, 'unique']]
            
        correct_column = pd.DataFrame(aux[column])

    if drop:
        df.drop(column, axis=1, inplace=True)
        
    df = df.merge(
        correct_column, 
        left_on=[
            prim_column,
        ], 
        right_on=[
            prim_column,
        ],
        how='left',
        suffixes=("_old", "_correct")
    ).copy()

    #print(df.shape)
    
    return df

def print_partitions(ddf:dd.core.DataFrame) -> None:
    for i in range(ddf.npartitions):
        print('Partion:', i)
        print(ddf.partitions[i].head())

def read_dask_dataframe(folder_path:str, folder_type:str, config:dict, add_meta:bool=False) -> dd.core.DataFrame:
    assert folder_path != '' 
    assert folder_type != '' 
    assert not config.empty
    
    ddf = None
    
    if folder_type == 'csv':
        # re read file
        ddf = dd.read_csv(
            urlpath=f'{folder_path}/{config.year}/{config.dataset}/{config.year}_{config.month:02d}_{config.monthname}_{config.dataset}.{folder_type}',
            blocksize='default',
            lineterminator=None,
            compression='infer',
            sample=256000,
            enforce=False,
            assume_missing=False,
            storage_options=None,
            include_path_column=False,
            header=0,
            dtype={'post_code': 'float64','street_number': 'object','street_name': 'object'}
        )
    else: 
        raise 'Not supported yet'
    
    if add_meta: 
        ddf._name = f'{config.year}-{config.month}'
        # we have one partion
        # TODO
        # ddf.divisions = (0, ddf.shape[0].compute()-1)
    
    return ddf

def read_dask_dataframes(folder_path:str, folder_type:str, input_dataset:str, years:List[int]) -> Dict[str, dd.core.DataFrame]:
    assert folder_path != '' 
    assert folder_type != ''
    assert input_dataset != ""
    
    data = dict()
    
    for year in tqdm(years):
        assert year >= 2018 and year <= 2023
        ddf_year_list = list()
        
        #print('--> ', year, input_dataset)
        
        config = pd.Series({
            'year':year,
            'dataset': input_dataset,
            'month': np.nan,
            'monthname': np.nan
        })
        
        for month, month_name in tqdm(i2m):
            config.month = month
            config.monthname = month_name
            #print('----> ', year, month, month_name, input_dataset)
            
            ddf_year_list.append(
                read_dask_dataframe(folder_path, folder_type, config)
            )
            
            #print('----> ', 'Done -------- ----------')
        
        data[year] = dd.concat(ddf_year_list, interleave_partitions=False)
        
        #print('--> ', 'Done -------- ----------')
        
    return data

def get_ddf_shape(ddf:dd.core.DataFrame):
    return ddf.shape[0].compute(), ddf.shape[1]

def get_column(df, column:str) -> pd.Series:
    assert column != ''
    assert df[column] is not None
    
    if type(df) == pd.DataFrame:
        return df[column]
    elif type(df) == dd.core.DataFrame: 
        return df[column].compute()
    
    raise Exception('Datatype not supported yet')
    
def get_column_value_counts(s) -> pd.Series:
    
    assert s is not None
    
    if type(s) == pd.Series:
        return s.value_counts()
    elif type(s) == dd.core.Series: 
        return s.value_counts().compute()
    
    raise Exception('Datatype not supported yet')
    

def scatter_columns(
    df, 
    col_x:str, 
    col_y:str, 
    col_z:str, 
    tail:bool=False, 
    xticks:np.ndarray=np.ndarray((0,0)), 
    yticks:np.ndarray=np.ndarray((0,0)),
    figsize:tuple=(20,25),
    count:int=5,
    label:str=None
) -> None:
    
    assert col_x != ''
    assert col_y != ''
    assert col_z != ''
    
    assert df[col_x] is not None
    assert df[col_y] is not None
    assert df[col_z] is not None
    
    plt.rcParams["figure.figsize"] = figsize
    
    counts = get_column_value_counts(df[col_z])
    
    print('value counts stats', {'max':counts.max(),'mean':counts.mean(),'median':counts.median(),'std':counts.std(),'min':counts.min()})
    
    if tail:
        keys = counts.tail(count).keys()
    else:    
        keys = counts.head(count).keys()
    
    colors = mpl.cm.rainbow(np.linspace(0, 1, len(keys)))

    label = f'-{label}' if label else ''
    
    for i, value in enumerate(tqdm(keys)):
        
        computed = df[df[col_z] == value].groupby([col_x])[col_y].mean().reset_index().compute().sort_values(by=col_x)
        
        x = get_column(computed, col_x)
        y = get_column(computed, col_y)
        
        plt.scatter(x, y, linewidths=True, label=f'{col_z}:{value}{label}', edgecolors=colors[i])
        plt.plot(x, y, linestyle='dashed', color='gray')
        
    if xticks.any():
        plt.xticks(xticks)
        
    if yticks.any():
        plt.yticks(yticks)
    
    plt.legend() # keys.astype(np.int)
    #plt.show()
    
# code to save checkpoint

def save_checkpoint(ddf:dd.core.DataFrame, config_year:dict):

    path_to_file = f'{config_year.path}/{config_year.year}/{config_year.dataset}'

    # DASK has so many issue it does replace files if already exists. For that reason it was needed to do this fix
    # deleting files before saving 
    # delete_dataset(path_to_file)

    os.system(f"mkdir -p {path_to_file}")
    
    ddf.to_csv(f'{path_to_file}/{config_year.year}_{config_year.dataset}_*.csv', index=False, mode='wt')
    
    print('checkpoint saved.')
    
def delete_dataset(path_to_dataset_folder:str):
    
    try:
        shutil.rmtree(path_to_dataset_folder)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (path_to_dataset_folder, e))

def load_checkpoint(config_year:dict) -> dd.core.DataFrame:
    
    path_files_year = f'{config_year.path}/{config_year.year}/{config_year.dataset}'

    if os.path.exists(f'{path_files_year}/{config_year.year}_{config_year.dataset}_00.csv'):

        ddf = dd.read_csv(
            f'{path_files_year}/{config_year.year}_{config_year.dataset}_*.csv', 
            dtype={
                'month': 'int64',
                'year': 'int64',
                'day': 'int64',
                'dayofweek': 'int64',
                'dayofyear': 'int64',
                'hour': 'int64',
                'timestamp': 'int64',
                'is_charging_station': 'int64',
                'is_installed': 'int64',
                'is_renting': 'int64',
                'is_returning': 'int64',
                'station_id': 'int64',
                'status': 'int64'
            }
        )
        
        print('checkpoint reloaded.')
    
        return ddf
    else:
        raise Exception('Files not found.')

def Cov(X, Y): 
    def _get_dvis(V):
        return [v - np.mean(V) for v in V]
    dxis = _get_dvis(X)
    dyis = _get_dvis(Y)
    return np.sum([x * y for x, y in zip(dxis, dyis)])/len(X)

def PearsonCorr(X, Y):
    assert len(X) == len(Y)
    return Cov(X, Y) / np.prod([np.std(V) for V in [X, Y]])

def list2rank(l):
    #l is a list of numbers
    # returns a list of 1-based index; mean when multiple instances
    return [np.mean([i+1 for i, sorted_el in enumerate(sorted(l)) if sorted_el == el]) for el in l]

def spearmanRank(X, Y):
    # X and Y are same-length lists
    return PearsonCorr(list2rank(X), list2rank(Y))

