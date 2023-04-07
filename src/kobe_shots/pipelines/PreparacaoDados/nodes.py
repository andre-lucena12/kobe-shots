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

    filtered_2fg = filter_2fg.dropna()
    filtered_3fg = filter_3fg.dropna()

    return filtered_2fg, filtered_3fg

def separe_data(data_2fg, test_size, random_state):

    X_without_scaler = data_2fg.drop('shot_made_flag', axis=1)
    y = data_2fg[['shot_made_flag']]

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X_without_scaler), 
                    columns = X_without_scaler.columns,
                    index = X_without_scaler.index)
    
    data_normalized = X.join(y)

    data_train, data_test = train_test_split(data_normalized,
                                            test_size = test_size, 
                                            random_state = random_state,
                                            stratify = data_normalized['shot_made_flag'])

    return data_train, data_test

def metrics_dataset(data_train, data_test):

    return{
        'Test_rows': {'value':data_test.shape[0], 'step':1},
        'Train_rows': {'value':data_train.shape[0], 'step':2}
    }

