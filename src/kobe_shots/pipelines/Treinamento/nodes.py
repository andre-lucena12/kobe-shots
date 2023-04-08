"""
This is a boilerplate pipeline 'Treinamento'
generated using Kedro 0.18.7
"""

import pandas as pd
from pycaret.classification import *
from sklearn.metrics import log_loss

def logistic_regression_train(data_train, data_test, target, random_seed, train_size, fold_strategy, fold):

    stp = setup(data= data_train, 
                target= target, 
                session_id= random_seed, 
                train_size= train_size,
                fold_strategy= fold_strategy,
                fold= fold,
                log_experiment= True)
    
    add_metric('logloss', 'Log Loss', log_loss)

    lr_model = create_model('lr', fold= fold)

    y_true = data_test['shot_made_flag']
    X_test = data_test.drop(columns='shot_made_flag')
    y_pred = predict_model(lr_model, data= X_test)['Label']

    score = log_loss(y_true, y_pred)

    return lr_model, {
        'Score_LogLoss_Lr_Model':{'value': score, 'step':1}
    }