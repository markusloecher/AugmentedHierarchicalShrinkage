import sys
sys.path.append("../arne/")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from aughs import ShrinkageClassifier, cross_val_shrinkage
from tqdm import trange
import joblib
import pandas as pd

# Get the data
X = pd.read_csv("OMICS.txt","\t")
X = np.array(X)
y = pd.read_csv("omics_target.txt","\t")
y = np.array(y).ravel()

# Compute importances for classical RF/DT
clf = RandomForestClassifier().fit(X, y)
FI_no_shrinkage = clf.feature_importances_

#shrink_modes = ["hs", "hs_entropy", "hs_log_cardinality", "hs_permutation"]
lambdas = [0., 0.1, 1.0, 10.0, 25.0, 50.0, 100.0]

# Hierarchical Shrinkage
shrink_mode = ["hs"]

# Create base classifier
hsc = ShrinkageClassifier(RandomForestClassifier())

# Perform grid search for best value of lambda
param_grid = {"shrink_mode": shrink_mode, "lmb": lambdas}

lmb_scores = cross_val_shrinkage(
    hsc,
    X,
    y,
    param_grid,
    n_splits=5,
    n_jobs=1,
    return_param_values=False,
    verbose=0,
)

best_idx = np.argmax(lmb_scores)
best_lmb = lambdas[best_idx]

# Get feature importances for best value of lambda
hsc.shrink_mode = shrink_mode
hsc.lmb = best_lmb
hsc.fit(X, y)
FI_hs = hsc.estimator_.feature_importances_