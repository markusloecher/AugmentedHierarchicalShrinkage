"""
    Experiment for Seminar with Random Forests

    :author: Anna Saranti
    :copyright: © 2023 HCI-KDD (ex-AI) group
    :date: 2023-02-15
"""

import copy
import os

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split

from TreeModelsFromScratch.RandomForest import RandomForest

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

########################################################################################################################
# [1.] Train test split ================================================================================================
########################################################################################################################
print("Train test split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

########################################################################################################################
# [2.] Fit regular RF model ============================================================================================
########################################################################################################################
# Fit regular RF model
rf = RandomForest(n_trees=100, treetype='classification', HShrinkage=False, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)                                         # predict class
y_pred_prob = rf.predict_proba(X_test)                              # Predict probabilites for class membership
roc_auc_performance = roc_auc_score(y_test, y_pred)                 # ROC AUC score on test set
print(f"ROC-AUC performance regular: {roc_auc_performance}")

########################################################################################################################
# [3.] Fit DT model with HS *without* lambda_multiplier ================================================================
# ######################################################################################################################
HS_lambda = 100.0
rf = RandomForest(treetype='classification',
                  HShrinkage=True,
                  HS_lambda=HS_lambda,
                  b_use_hs_lambda_multiplier=False,
                  random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)                                         # predict class
y_pred_prob = rf.predict_proba(X_test)                              # Predict probabilites for class membership
roc_auc_performance = roc_auc_score(y_test, y_pred)                 # ROC AUC score on test set
# mi_hs_only = mutual_info_classif(y_test.values.reshape(-1, 1), y_pred.reshape(-1, 1))
print(f"ROC-AUC performance with HS *without* lambda_multiplier: {roc_auc_performance}")

########################################################################################################################
# [3.] Fit DT model with HS *with* lambda_multiplier ================================================================
# ######################################################################################################################
rf = RandomForest(treetype='classification',
                  HShrinkage=True,
                  HS_lambda=HS_lambda,
                  b_use_hs_lambda_multiplier=True,
                  random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)                                         # predict class
y_pred_prob = rf.predict_proba(X_test)                              # Predict probabilites for class membership
roc_auc_performance = roc_auc_score(y_test, y_pred)                 # ROC AUC score on test set
# mi_hs_lambda = mutual_info_classif(y_test.values.reshape(-1, 1), y_pred.reshape(-1, 1))
print(f"ROC-AUC performance with HS *with* lambda_multiplier: {roc_auc_performance}")
