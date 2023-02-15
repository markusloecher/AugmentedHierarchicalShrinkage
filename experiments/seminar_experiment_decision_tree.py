"""
    Experiment for Seminar with Decision Trees

    :author: Anna Saranti
    :copyright: © 2023 HCI-KDD (ex-AI) group
    :date: 2023-02-14
"""

from TreeModelsFromScratch.DecisionTree import DecisionTree
from TreeModelsFromScratch.RandomForest import RandomForest
from TreeModelsFromScratch.SmoothShap import smooth_shap, GridSearchCV_scratch, cross_val_score_scratch

import copy
import shap
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
import os

from TreeModelsFromScratch.RandomForest import RandomForest
from TreeModelsFromScratch.datasets import DATASETS_CLASSIFICATION, DATASET_PATH
from sklearn.model_selection import train_test_split
from imodels import get_clean_dataset
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

########################################################################################################################
# [0.] Load the data ===================================================================================================
########################################################################################################################
# [0.1.] Titanic data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Load the data")
data_path = os.path.join("data", "titanic")

# Load and clean
data = pd.read_csv(os.path.join(data_path, "titanic_train.csv"))
data = data[data["Age"].notnull()]                              # filter rows which are nan
data["Sex"] = pd.get_dummies(data["Sex"])["female"]             # dummy code sex (1==Female)

# Create X and y
X = data[['Age', 'Pclass', 'Sex', 'PassengerId']]
y = data["Survived"].astype("float")

"""
# [0.2.] German credit ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data_path = os.path.join("data", "german_credit")
data = pd.read_csv(os.path.join(data_path, "german.tsv"), sep='\t')
data_column_names = list(data.columns.values)
# print(f"Data column names: {data_column_names}")

nan_values_nr = data.isnull().sum().sum()
print(f"Nr. of NaN values: {nan_values_nr}")

data_column_names_X = copy.deepcopy(data_column_names)
data_column_names_X.remove("target")
X = data[data_column_names_X]
y = data["target"].values
"""
########################################################################################################################
# [1.] Train test split ================================================================================================
########################################################################################################################
print("Train test split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

########################################################################################################################
# [2.] Fit regular DT model ============================================================================================
########################################################################################################################
dt = DecisionTree(treetype='classification', HShrinkage=False, random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)                                         # predict class
y_pred_prob = dt.predict_proba(X_test)                              # Predict probabilites for class membership
roc_auc_performance = roc_auc_score(y_test, y_pred)                 # ROC AUC score on test set

# mi_regular = mutual_info_classif(y_test.values.reshape(-1, 1), y_pred.reshape(-1, 1))
print(f"ROC-AUC performance regular: {roc_auc_performance}")

########################################################################################################################
# [3.] Fit DT model with HS *without* lambda_multiplier ================================================================
# ######################################################################################################################
HS_lambda = 10.0
dt = DecisionTree(treetype='classification',
                  HShrinkage=True,
                  HS_lambda=HS_lambda,
                  b_use_hs_lambda_multiplier=False,
                  random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)                                         # predict class
y_pred_prob = dt.predict_proba(X_test)                              # Predict probabilites for class membership
roc_auc_performance = roc_auc_score(y_test, y_pred)                 # ROC AUC score on test set
# mi_hs_only = mutual_info_classif(y_test.values.reshape(-1, 1), y_pred.reshape(-1, 1))
print(f"ROC-AUC performance with HS *without* lambda_multiplier: {roc_auc_performance}")

########################################################################################################################
# [3.] Fit DT model with HS *with* lambda_multiplier ================================================================
# ######################################################################################################################
dt = DecisionTree(treetype='classification',
                  HShrinkage=True,
                  HS_lambda=HS_lambda,
                  b_use_hs_lambda_multiplier=True,
                  random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)                                         # predict class
y_pred_prob = dt.predict_proba(X_test)                              # Predict probabilites for class membership
roc_auc_performance = roc_auc_score(y_test, y_pred)                 # ROC AUC score on test set
# mi_hs_lambda = mutual_info_classif(y_test.values.reshape(-1, 1), y_pred.reshape(-1, 1))
print(f"ROC-AUC performance with HS *with* lambda_multiplier: {roc_auc_performance}")

