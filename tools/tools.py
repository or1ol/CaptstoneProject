import os

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
    
    show_column_counts(df[column])

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

def read_status_estacion_mes(config:dict):
    data_df = pd.read_csv(f'../dades/{config.year}/{config.datafrom}/{config.year}_{config.month:02d}_{config.monthname}_{config.datafrom}.csv')

    intial_size = data_df.shape[0]
    print(data_df.shape)

    # change column to one hot enconding
    data_df['is_charging_station'] = data_df.is_charging_station.astype(np.int)

    # STATUS = IN_SERVICE=En servei, CLOSED=Tancada, MAINTENANCE=installed but closed for MAINTENANCE, PLANNED=not installed and closed
    # replace IN_SERVICE with 1 and CLOSED with 0 
    data_df['status'].replace(
        to_replace=['IN_SERVICE', 'OPEN', 'OPN', 'CLS', 'CLOSED', 'NOT_IN_SERVICE', 'MAINTENANCE', 'PLANNED'],                       
        value=[0, 0, 0, 1, 1, 1,  2, 3], inplace=True)
    
    data_df.loc[data_df.last_reported.isna(), 'last_reported'] = data_df.loc[data_df.last_reported.isna(), 'last_updated']

    # will remove the duplicate for last reported for all stations in the dataset
    data_df = remove_duplicates_all(data_df.copy(), 'last_reported')

    # convert timestamps of last_updated
    data_df = convert_timestamp(data_df.copy(), ['last_updated'], sort=True, add=True)

    # convert timestamps to multimple of 60
    data_df = timestamp_multipleof(
        devide_by=config.devide_by, 
        column='minutes_last_updated_date',
        df=data_df.copy(), 
        new_column='last_updated', 
        year_column='year_last_updated_date',
        month_column='month_last_updated_date',
        day_column='dayofmonth_last_updated_date',
        hour_column='hour_last_updated_date',
        minutes_column='minutes_last_updated_date'
    )
    
    # print(data_df.minutes_last_updated_date.value_counts())
    data_df.drop(['minutes_last_updated_date'], axis=1, inplace=True)

    ### will remove the duplicate for last reported for all stations in the dataset
    data_df = remove_duplicates_all(data_df.copy(), 'last_updated')
    
    print(data_df.shape)
    print('removed:', intial_size-data_df.shape[0])
    
    data_df.reset_index(drop=True, inplace=True)

    data_df.drop(['ttl'], axis=1, inplace=True)

    # save checkpoint
    data_df.to_csv(f'../dades/{config.year}/{config.dataset}/{config.year}_{config.month:02d}_{config.monthname}_{config.dataset}.csv', index=False)


def read_status_informacio_mes(config:dict) -> dd.core.DataFrame:

    data_df = pd.read_csv(f'../dades/{config.year}/{config.datafrom}/{config.year}_{config.month:02d}_{config.monthname}_{config.datafrom}.csv')

    intial_size = data_df.shape[0]
    print(data_df.shape)
    
    # drop not needed columns
    # data_df.drop(['nearbyStations', 'cross_street'], axis=1, inplace=True)

    data_df.loc[data_df.altitude.isin(['0.1', 'nan', np.nan]), 'altitude'] = '0'
    data_df.altitude = data_df.altitude.astype(np.int).astype(str)

    cond = (~data_df.altitude.isin([str(x) for x in range(200)] + [np.nan]))
    print(data_df[cond].shape)
    # 485 row does not have 0 in the altitud column
    # capacity is filled with values 1 to fix this we need to shift the data 

    # Fix data 
    data_df.loc[cond, ['capacity']] = data_df[cond].post_code
    data_df.loc[cond, ['post_code']] = data_df[cond].address
    data_df.loc[cond, ['address']] = data_df[cond].altitude
    data_df.loc[cond, ['altitude']] = '0'
    data_df.altitude.fillna('0', inplace=True)

    # post code is wrong need fixing using long & lat. 
    # can be fixed using post code data from old dataset after the merge
    data_df['post_code'] = '0'

    data_df = convert_timestamp(data_df.copy(), ['last_updated'], sort=True, add=True)

    # convert timestamps to multimple of 3
    data_df = timestamp_multipleof(
        devide_by=config.devide_by, 
        column='minutes_last_updated_date',
        df=data_df.copy(), 
        new_column='last_updated', 
        year_column='year_last_updated_date',
        month_column='month_last_updated_date',
        day_column='dayofmonth_last_updated_date',
        hour_column='hour_last_updated_date',
        minutes_column='minutes_last_updated_date'
    )

    # drop not needed columns
    data_df.drop(
        [
            'year_last_updated_date', 'month_last_updated_date',
            'week_last_updated_date', 'dayofweek_last_updated_date',
            'dayofmonth_last_updated_date', 'dayofyear_last_updated_date',
            'hour_last_updated_date', 'minutes_last_updated_date'
        ],
        axis=1,
        inplace=True
    )

    data_df['physical_configuration'].replace(to_replace=['REGULAR', 'BIKE','BIKESTATION', 'BIKE-ELECTRIC', 'ELECTRICBIKESTATION'], value=[0, 0, 0, 1, 1], inplace=True)

    # create mew column of last reported and last updated 
    data_df['street_name'] = data_df.apply(
        lambda x: " ".join(re.findall("[a-zA-Z]+", x['name'])),
        axis=1
    )

    def lambda_fun(name):
        ret = 'nan'
        try:
            ret = re.findall("\d+$", name)[0]
        except:
            ret = 'nan'

        return ret

    # create mew column of last reported and last updated 
    data_df['street_number'] = data_df.apply(
        lambda x: lambda_fun(x['name']),
        axis=1
    )

    # we don't have this column anywhere in the new dataset so it got removed
    data_df.drop(['address', 'name'], axis=1, inplace=True)

    ### will remove the duplicate for last reported for all stations in the dataset
    data_df = remove_duplicates_all(data_df.copy(), 'last_updated')
    
    print(data_df.shape)
    print('removed:', intial_size-data_df.shape[0])
    
    data_df.reset_index(drop=True, inplace=True)

    data_df.drop(['ttl'], axis=1, inplace=True)

    # save checkpoint
    data_df.to_csv(f'../dades/{config.year}/{config.dataset}/{config.year}_{config.month:02d}_{config.monthname}_{config.dataset}.csv', index=False)


def get_file_length(config:dict):
    data_df = pd.read_csv(
        filepath_or_buffer=f'../dades/{config.year}/{config.datafrom}/{config.year}_{config.month:02d}_{config.monthname}_{config.datafrom}.csv',
        header=0,
        low_memory=False,
    )
    return data_df.shape
    
def read_informacion_estacion_anual(input_dataset:str, year:int):
    assert input_dataset != ""
    assert year >= 2018 and year <= 2023

    config = pd.Series({
        'devide_by':60,
        'year':year,
        'datafrom': input_dataset,
        'dataset': f'{input_dataset}_CLEAN',
        'ttl': 30,
        'month': np.nan,
        'monthname': np.nan
    })

    os.system(f"mkdir -p ../dades/{config.year}/{config.dataset}")

    for month, month_name in i2m:
        config.month = month
        config.monthname = month_name
        print(year, month, month_name, input_dataset)
        if not os.path.exists(f'../dades/{config.year}/{config.dataset}/{config.year}_{config.month:02d}_{config.monthname}_{config.dataset}.csv'):
            if input_dataset == 'BicingNou_ESTACIONS':
                read_status_estacion_mes(config)
            elif input_dataset == 'BicingNou_INFORMACIO':
                read_status_informacio_mes(config)
            # TODO add elif para cada dataset que queramso anadir en el futuro ()
        else:
            print('found file with shape equal to: ', get_file_length(config))
            
        print('Done -------- ----------')

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
            header=0
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
