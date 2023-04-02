"""
This is a boilerplate pipeline 'PreparacaoDados'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_data, filter_data

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
        )
    ])
