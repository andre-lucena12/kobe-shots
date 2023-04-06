"""
This is a boilerplate pipeline 'PreparacaoDados'
generated using Kedro 0.18.7
"""

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(data):
    
    data = data
    return data

def filter_data(kobe_dataset):

    separador = '2PT Field Goal'
    filter_2fg_partial = kobe_dataset[kobe_dataset['shot_type'] == separador]
    filter_3fg_partial = kobe_dataset[kobe_dataset['shot_type'] != separador]

    filter_2fg = filter_2fg_partial[['lat',
                                     'lon',
                                     'minutes_remaining',
                                     'period',
                                     'playoffs',
                                     'shot_distance',
                                     'shot_made_flag']]
    filter_3fg = filter_3fg_partial[['lat',
                                     'lon',
                                     'minutes_remaining',
                                     'period',
                                     'playoffs',
                                     'shot_distance',
                                     'shot_made_flag']]

    filter_2fg = filter_2fg.dropna()
    filter_3fg = filter_3fg.dropna()

    return filter_2fg, filter_3fg

def separe_data(data_2fg, test_size, random_state):

    X_without_scaler = data_2fg.drop('shot_made_flag', axis=1)
    y = data_2fg[['shot_made_flag']]

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X_without_scaler), columns = X_without_scaler.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = test_size, 
                                                        random_state = random_state)

    return X_train, X_test, y_train, y_test

def metrics_dataset(X_train, X_test):

    rows_train, cols_train = X_train.shape()
    rows_test, cols_test = X_test.shape()

    return rows_train, cols_train, rows_test, cols_test

