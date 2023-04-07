"""
This is a boilerplate pipeline 'Treinamento'
generated using Kedro 0.18.7
"""

import pandas as pd
from pycaret.classification import *

def logistic_regression_train(X_train, y_train):

    clf_model = setup(data=pd.merge(X_train, y_train))

    return