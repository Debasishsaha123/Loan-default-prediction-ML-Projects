import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import os
import warnings

import os

import kaggle

# kaggle.api.authenticate()
# kaggle.api.dataset_download_files('braindeadcoder/lending-club-data', path='.',unzip=True)

# # Configurações Iniciais
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings('ignore')

df=pd.read_csv('loan_data.csv')
print(df.info())

