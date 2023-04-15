"""
This is a boilerplate pipeline 'Treinamento'
generated using Kedro 0.18.7
"""

import pandas as pd
from pycaret.classification import *
from sklearn.metrics import log_loss, f1_score, classification_report

def logistic_regression_train(data_train, data_test, target, random_seed, train_size, fold_strategy, fold):

    stp = setup(data= data_train, 
                target= target, 
                session_id= random_seed, 
                train_size= train_size,
                fold_strategy= fold_strategy,
                fold= fold,
                log_experiment= True,
                experiment_name= 'tuned_lr_model_logloss')
    
    add_metric('logloss', 'Log Loss', log_loss)

    lr_model = create_model('lr', fold= fold)
    tuned = tune_model(lr_model)

    y_true = data_test['shot_made_flag']
    X_test = data_test.drop(columns='shot_made_flag')
    y_pred_proba = tuned.predict_proba(X_test)
    y_pred = predict_model(tuned, data=data_test)['prediction_label']

    report = classification_report(y_true, y_pred, output_dict= True)

    precision = report['1.0']['precision']
    recall = report['1.0']['recall']
    f1_score = report['1.0']['f1-score']
    acuracia = report['accuracy']
    logloss_score = log_loss(y_true, y_pred_proba)
    
    metricas_df = pd.DataFrame({
        'LogLoss_Tuned_Lr_Model':{'value': logloss_score, 'step':1},
        'Precision':{'value':precision, 'step':2},
        'Recall':{'value':recall, 'step':3},
        'F1_Score':{'value':f1_score, 'step':4},
        'Accuracy':{'value':acuracia, 'step':5}
    })

    return tuned, {
        'LogLoss_Tuned_Lr_Model':{'value': logloss_score, 'step':1},
        'Precision':{'value':precision, 'step':2},
        'Recall':{'value':recall, 'step':3},
        'F1_Score':{'value':f1_score, 'step':4},
        'Accuracy':{'value':acuracia, 'step':5}
    }, metricas_df

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

    best_model = compare_models()
    tuned_best = tune_model(best_model)

    y_true = data_test['shot_made_flag']
    X_test = data_test.drop(columns='shot_made_flag')
    y_pred_proba = tuned_best.predict_proba(X_test)
    y_pred = predict_model(tuned_best, data=data_test)['prediction_label']

    report_best = classification_report(y_true, y_pred, output_dict= True)

    precision = report_best['1.0']['precision']
    recall = report_best['1.0']['recall']
    f1_score = report_best['1.0']['f1-score']
    acuracia = report_best['accuracy']
    logloss_score = log_loss(y_true, y_pred_proba)

    metricas_best_df = pd.DataFrame({'LogLoss_Best_Model':{'value': logloss_score, 'step':1},
                            'Precision_Best_Model':{'value':precision, 'step':2},
                            'Recall_Best_Model':{'value':recall, 'step':3},
                            'F1_Score_Best_Model':{'value':f1_score, 'step':4},
                            'Accuracy_Best_Model':{'value':acuracia, 'step':5}})

    return tuned_best, {'LogLoss_Best_Model':{'value': logloss_score, 'step':1},
                        'Precision_Best_Model':{'value':precision, 'step':2},
                        'Recall_Best_Model':{'value':recall, 'step':3},
                        'F1_Score_Best_Model':{'value':f1_score, 'step':4},
                        'Accuracy_Best_Model':{'value':acuracia, 'step':5}}, metricas_best_df
    