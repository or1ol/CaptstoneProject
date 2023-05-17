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

# study of skewness of the data population
def skewness(df:pd.DataFrame, column:str):
    """
    Mesures of asymetry: 
    Negative deviation indicates that the destribution skews left. The skewness of a normal distrubtion is zero. And any symetric data must have skewness equal to zero.
    The alternative to this is by lookig into the relationship between the mean and the median.
    """
    assert df[column] is not None
    assert column != ''
    
    result = 0
    series = df[column]
    data_mean = series.mean()
    data_std = series.std()
    data_count = len(series)
    
    for i in series:
        result += ((i - data_mean) * (i - data_mean) * (i - data_mean))
    result /= (data_count * data_std * data_std * data_std)
    
    return result

# Pearson median skewness coefficient
def pearson(df:pd.DataFrame, column:str):
    """
    is an alternative to skewness coefficient
    """
    assert df[column] is not None
    assert column != ''
    
    result = 0
    series = df[column]
    data_mean = series.mean()
    data_median = series.median()
    data_std = series.std()
    data_count = len(series)
    
    result = 3*(data_mean - data_median)*data_std
    
    return result


def get_features_nans(df:pd.DataFrame):
    result = (df.isna().sum()/df.shape[0])*100
    return result[result > 0].to_dict()

def get_features_zero(df:pd.DataFrame):
    result = (df.isin([0]).sum()/df.shape[0])*100
    return result[result > 0].to_dict()

def get_nans_counts(df:pd.DataFrame, column:str, mean_of_column:str, mean_of_value):
    return df[df[mean_of_column] == mean_of_value][column].isna().sum()

def get_columns_nunique(df: pd.DataFrame, cat_only:bool=False, num_only:bool=False) -> dict:
    if cat_only:
        cat_col = df.select_dtypes(include=['object']).columns
        return {col:len(set(df[col])) for col in cat_col}
    if num_only:
        num_col = df.select_dtypes(exclude=['object']).columns
        return {col:len(set(df[col])) for col in num_col}
    return {col:df[col].nunique() for col in df.columns}

def get_columns_unique(df: pd.DataFrame, cat_only:bool=False, num_only:bool=False) -> dict:
    if cat_only:
        cat_col = df.select_dtypes(include=['object']).columns
        return {col:len(set(df[col])) for col in cat_col}
    if num_only:
        num_col = df.select_dtypes(exclude=['object']).columns
        return {col:len(set(df[col])) for col in num_col}
    return {col:df[col].unique() for col in df.columns}

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

def show_column_counts(df:pd.DataFrame, column:str):
    assert column != ''
    assert df[column] is not None
    
    
    fig, axs = plt.subplots(3, 2, figsize=(20,10))
    axs[0][0].hist(df[column], label=f'{column} hist',bins=40)
    axs[0][0].set_xlabel('values')
    axs[0][0].set_ylabel('counts')
    axs[0][0].set_title('')

    axs[0][1].scatter(df[column].index, df[column].values, label=f'{column} scatter')
    axs[0][1].set_xlabel('index')
    axs[0][1].set_ylabel('values')
    axs[0][1].set_title('')
    
    axs[1][0].scatter(df[column].value_counts().index, df[column].value_counts().values,label=f'{column} counts')
    axs[1][0].set_xlabel('values')
    axs[1][0].set_ylabel('counts')
    axs[1][0].set_title('')
    
    axs[1][1].hist(df[column].value_counts(),label=f'{column} counts', bins=df[column].value_counts().shape[0])
    axs[1][1].set_xlabel('counts')
    axs[1][1].set_ylabel('values')
    axs[1][1].set_title('')
    
    
    axs[2][0].hist(df[column], density=True, histtype='step', cumulative=True,  linewidth=3.5, bins=30, color=sns.desaturate("indianred", .75))
    axs[2][0].set_xlabel('values')
    axs[2][0].set_ylabel('counts')
    axs[2][0].set_title('')
    
    axs[2][1].boxplot(df[column])
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
            aux[cat_col] = candidates.dropna()[cat_col].value_counts().index[0]
        
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

# This function works only for data of one station
def remove_duplicates_dask(df:pd.DataFrame, column:str, station_id:int, config:dict) -> pd.DataFrame:
    assert column != ''
    assert df[column] is not None
    
    df_dd = dd.from_pandas(df, npartitions=1, name=f'{config.year}-{config.month}-{station_id}')
    
    aux = df_dd[column].value_counts()
    repeated_data = aux[aux > 1].compute()
    
    for value in repeated_data.index:
        index = df_dd[column] == value
        aux = df_dd[index] # taking only the ones with ttl bigger then 10

        candidates = aux[aux['ttl'] >= 10]
        candidates = candidates if candidates.shape[0].compute() > 1 else aux

        cat_cols = df_dd.select_dtypes(include=['object']).columns
        num_cols = df_dd.select_dtypes(exclude=['object']).columns

        aux = candidates.mean().compute() #.round().astype(np.int)

        for cat_col in cat_cols:
            aux[cat_col] = candidates.dropna()[cat_col].value_counts().compute().index[0]

        assert df_dd.shape[1] == aux.shape[0]

        df_dd = df_dd[~index]
        df_dd = df_dd.append(aux)
        
    # reorder the list
    df_dd = df_dd.reset_index(drop=True)

    return df_dd.compute()

def remove_duplicates_all_dask(df:pd.DataFrame, column:str, config:dict) -> pd.DataFrame:
    assert column != ''
    assert df[column] is not None
    
    result = {}
    
    for station_id in tqdm(df.station_id.unique().tolist()):
        df_s = df[df.station_id == station_id]
        df_s = remove_duplicates_dask(df_s.copy(), column, station_id, config)
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