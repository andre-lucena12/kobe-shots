"""
This is a boilerplate pipeline 'Treinamento'
generated using Kedro 0.18.7
"""

import pandas as pd
from pycaret.classification import *
from sklearn.metrics import log_loss, f1_score

def logistic_regression_train(data_train, data_test, target, random_seed, train_size, fold_strategy, fold):

    stp = setup(data= data_train, 
                target= target, 
                session_id= random_seed, 
                train_size= train_size,
                fold_strategy= fold_strategy,
                fold= fold,
                log_experiment= True,
                experiment_name= 'lr_model_logloss')
    
    add_metric('logloss', 'Log Loss', log_loss)

    lr_model = create_model('lr', fold= fold, return_train_score= True)

    y_true = data_test['shot_made_flag']
    X_test = data_test.drop(columns='shot_made_flag')
    y_pred = lr_model.predict_proba(X_test)

    score = log_loss(y_true, y_pred)

    return lr_model, {
        'Score_LogLoss_Lr_Model':{'value': score, 'step':1}
    }

def best_model_train(data_train, data_test, target, random_seed, train_size, fold_strategy, fold):

    stp = setup(data= data_train, 
                target= target, 
                session_id= random_seed, 
                train_size= train_size,
                fold_strategy= fold_strategy,
                fold= fold,
                log_experiment= True,
                experiment_name= 'best_model')
    
    add_metric('logloss', 'Log Loss', log_loss)
    add_metric('f1_score', 'F1 Score', f1_score)

    best_model = compare_models(fold= fold, return_train_score=True, sort= 'f1_score')
    
    