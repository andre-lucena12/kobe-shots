"""
This is a boilerplate pipeline 'Treinamento'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import logistic_regression_train, best_model_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func= logistic_regression_train,
            inputs= ['data_train', 
                     'data_test', 
                     'params:target', 
                     'params:random_seed', 
                     'params:train_size',
                     'params:fold_strategy',
                     'params:fold'],
            outputs= ['lr_model', 
                      'metricas_lr',
                      'metricas_df'],
            name= 'logistic_regression_train'
        ),
        node(
            func= best_model_train,
            inputs= ['data_train', 
                     'data_test', 
                     'params:target', 
                     'params:random_seed', 
                     'params:train_size',
                     'params:fold_strategy',
                     'params:fold'],
            outputs=['best_model', 
                     'metricas_best',
                     'metricas_best_df'],
            name= 'best_model_train'
        )
    ])
