"""
    Experiment for Seminar

    :author: Anna Saranti
    :copyright: © 2023 HCI-KDD (ex-AI) group
    :date: 2023-02-14
"""

from TreeModelsFromScratch.DecisionTree import DecisionTree
from TreeModelsFromScratch.RandomForest import RandomForest
from TreeModelsFromScratch.SmoothShap import smooth_shap, GridSearchCV_scratch, cross_val_score_scratch

import shap
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

from TreeModelsFromScratch.RandomForest import RandomForest
from TreeModelsFromScratch.datasets import DATASETS_CLASSIFICATION, DATASET_PATH
from sklearn.model_selection import train_test_split
from imodels import get_clean_dataset

########################################################################################################################
# [0.] Load the data ===================================================================================================
########################################################################################################################
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

########################################################################################################################
# [2.] Fit regular DT model ============================================================================================
########################################################################################################################
dt = DecisionTree(treetype='classification', HShrinkage=False, random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)                                         # predict class
y_pred_prob = dt.predict_proba(X_test)                              # Predict probabilites for class membership
roc_auc_performance = roc_auc_score(y_test, y_pred)                 # ROC AUC score on test set
print(f"ROC-AUC performance regular: {roc_auc_performance}")

########################################################################################################################
# [3.] Fit fancy DT model ==============================================================================================
########################################################################################################################
dt = DecisionTree(treetype='classification', HShrinkage=True, HS_lambda=100.0, random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)                                         # predict class
y_pred_prob = dt.predict_proba(X_test)                              # Predict probabilites for class membership
roc_auc_performance = roc_auc_score(y_test, y_pred)                 # ROC AUC score on test set
print(f"ROC-AUC performance mit fancy shrinkage: {roc_auc_performance}")


"""
mdi_feature = pd.DataFrame(rf.feature_importances_.reshape(1, 4), columns=rf.feature_names)
print("MDI feature:")
print(mdi_feature)

# Access first tree model in ensemble
dec_tree = rf.trees[0]
print(type(dec_tree))

print("Maximum depth of the fitted tree:", dec_tree.max_depth_)
print("Number of nodes in the fitted tree:", dec_tree.n_nodes)

print(f"Feature importances: {dec_tree.feature_importances_}")
print(f"Feature names: {dec_tree.feature_names}")

# Examine first node (root node)
print(dec_tree.node_id_dict.get(0))    # The keys in the dict are the corresponding node id

# Example of 1st entry in ode list (root node)
dec_tree.node_list[0]

# Explain decision path for 1st observation from test set
dec_tree.explain_decision_path(X_test.iloc[0])

# Explain decision path for 1st and 2nd observation from test set
dec_tree.explain_decision_path(X_test.iloc[:2])

# Instantiate and fit RF model with HS applied post-hoc
rf_hs = RandomForest(n_trees=100, treetype='classification', HShrinkage=True, HS_lambda=5, random_state=42)
rf_hs.fit(X_train, y_train)
y_pred = rf_hs.predict(X_test)
roc_auc_score(y_test, y_pred)   # ROC AUC score on test set

cv = 3
grid = {"HS_lambda": [0.1, 1, 10, 25, 50, 100]} # The key of the dict has to match the attribute name in the RF model

# Instantiate model and apply GridSearch to determine best lambda
rf_hs = RandomForest(n_trees=25, treetype="classification", random_state=42, HShrinkage=True)
# returns fitted model and stores results as dict
grid_cv_HS = GridSearchCV_scratch(rf_hs, grid, X_train, y_train, cv=cv, scoring_func=roc_auc_score)
print("Best lambda:", rf_hs.HS_lambda)

# GridSearchCV_scratch returns fitted model and stores results as dict
print(grid_cv_HS)

########################################################################################################################
# RF with AugHS smSHAP applied =========================================================================================
########################################################################################################################

rf_aug_smSH = RandomForest(n_trees=100,
                           treetype='classification',
                           HS_lambda=5,
                           oob_SHAP=True,
                           HS_smSHAP=True,
                           random_state=42)
rf_aug_smSH.fit(X_train, y_train)
y_pred = rf_aug_smSH.predict(X_test)
roc_auc_rf_aug_smSH = roc_auc_score(y_test, y_pred)     # ROC AUC score on test set
print(f"roc_auc_rf_aug_smSH: {roc_auc_rf_aug_smSH}")

# Fit regular RF model
rf = RandomForest(n_trees=100, treetype='classification', oob_SHAP=True, random_state=42)
rf.fit(X_train, y_train)

# Apply AugHS smSHAP post hoc on fitted RF model
rf.apply_smSHAP_HS(HS_lambda=5)
y_pred = rf.predict(X_test)
roc_auc_score(y_test, y_pred) # ROC AUC score on test set

########################################################################################################################
# RF with AugHS MSE applied ============================================================================================
########################################################################################################################
rf_aug_mse = RandomForest(n_trees=100,
                          treetype='classification',
                          HS_lambda=5,
                          HS_nodewise_shrink_type="MSE_ratio",
                          random_state=42)
rf_aug_mse.fit(X_train, y_train)
y_pred = rf_aug_mse.predict(X_test)
rf_aug_roc_auc = roc_auc_score(y_test, y_pred)   # ROC AUC score on test set

# Raw SHAP values of RF model
export_model_rf = rf.export_forest_for_SHAP()
explainer_rf = shap.TreeExplainer(export_model_rf)
shap_vals_rf_raw = explainer_rf.shap_values(X_train, y_train)

# Create SHAP summary plot
shap.summary_plot(shap_vals_rf_raw, X_train, X_train.columns, show=False)

# Compute smooth SHAP values -------------------------------------------------------------------------------------------
# Fit regular RF model with oob_SHAP=True
rf = RandomForest(n_trees=100, treetype='classification', oob_SHAP=True, random_state=42)
rf.fit(X_train, y_train)

# The inbag and oob SHAP values are stored as class attributes of the RF model object
print(f"Inbag SHAP values: {rf.inbag_SHAP_values}")     # Inbag SHAP values
print(f"OOB SHAP values: {rf.oob_SHAP_values}")         # OOB SHAP values

# Compute smooth SHAP values
smshap_vals, abs_mean_smshap, coefs_smshap = smooth_shap(rf.inbag_SHAP_values, rf.oob_SHAP_values)

# Smooth SHAP coefficient for each feature
print("Smooth SHAP coefficient for each feature")
smooth_shap_coeff_for_each_feature = pd.DataFrame(np.array([coefs_smshap]), columns=X_train.columns)
print(smooth_shap_coeff_for_each_feature)
"""
