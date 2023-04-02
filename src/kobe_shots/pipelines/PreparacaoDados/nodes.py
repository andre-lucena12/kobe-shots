"""
This is a boilerplate pipeline 'PreparacaoDados'
generated using Kedro 0.18.7
"""

import pandas as pd

def load_data(data):
    
    data = data
    return data

def filter_data(kobe_dataset):

    separador = '2PT Field Goal'
    filter_2fg = kobe_dataset[kobe_dataset['shot_type'] == separador]
    filter_3fg = kobe_dataset[kobe_dataset['shot_type'] != separador]

    filter_2fg = filter_2fg.dropna()
    filter_3fg = filter_3fg.dropna()

    return filter_2fg, filter_3fg


