# Scikit-Learn-compatible implementation of Augmented Hierarchical Shrinkage
This directory contains an implementation of Augmented Hierarchical Shrinkage in the [aughs](aughs) directory that is compatible with Scikit-Learn. It exports 4 classes:
- `AugHSTreeClassifier`
- `AugHSTreeRegressor`
- `AugHSForestClassifier`
- `AugHSForestRegressor`

These correspond to Scikit-Learn's `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier` and `RandomForestRegressor`, respectively. Each class has `fit()`, `predict()` and `predict_proba()` methods that work in much the same way as in the normal Scikit-Learn estimators.

The key differences are given by the arguments in the constructor:
- `base_estimator: sklearn.base.BaseEstimator:` this is the around which the hierarchical shrinkage is "wrapped". By default, this is a decision tree with `max_leaf_nodes=20` or a random forest with `n_estimators=100`.
- `lmb: float:` this is the value for lambda in hierarchical shrinkage.
- `shrink_mode: str:` this is the type of augmented hierarchical shrinkage to be used. There are 4 options:
    - `"hs":` "normal" (non-augmented) hierarchical shrinkage
    - `"hs_entropy":` multiplies lambda with the entropy of the split feature in the parent node
    - `"hs_entropy_2":` multiplies the entire regularization fraction with the entropy of the split feature in the parent node
    - `"hs_log_cardinality":` multiplies lambda with the log of the cardinality of the split feature in the parent node

Usage examples are given in the following notebooks:
- `aughs_cv.ipynb:` uses sklearn's `GridSearchCV` to optimize the `lmb` hyperparameter on the titanic dataset
- `titanic_mdi.ipynb:` demonstrates the influence of different types of hierarchical shrinkage on the MDI importance on the titanic dataset