"""
This is a boilerplate pipeline 'PreparacaoDados'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_data, filter_data, separe_data, metrics_dataset

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func = load_data,
            inputs = 'data',
            outputs = 'kobe_dataset',
            name = 'load_data',
        ),
        node(
            func = filter_data,
            inputs =  'kobe_dataset',
            outputs = ['2fg_dataset','3fg_dataset'],
            name = 'filter_data'
        ),
        node(
            func = separe_data,
            inputs = ['2fg_dataset','params:test_size','params:random_state'],
            outputs = ['data_train', 'data_test'],
            name = 'separe_data'
        ),
        node(
            func = metrics_dataset,
            inputs = ['data_train','data_test'],
            outputs = 'train_test_metrics',
            name = 'metrics_dataset'
        )
    ])
