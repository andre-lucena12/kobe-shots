"""
This is a boilerplate pipeline 'Treinamento'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import logistic_regression_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func= logistic_regression_train,
            inputs= ['data_train', 
                     'data_test', 
                     'param:target', 
                     'param:random_state', 
                     'param:train_size',
                     'param:fold_strategy',
                     'param:fold'],
            outputs= ['lr_model', 'score_logloss'],
            name= 'logistic_regression_train'
        )
    ])
