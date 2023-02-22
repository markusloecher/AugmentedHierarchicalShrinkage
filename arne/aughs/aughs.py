import scipy
import numpy as np
import pandas as pd
from typing import Optional, Union, List
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from numpy import typing as npt
from sklearn.utils.validation import check_X_y


def _check_fit_arguments(X, y, feature_names):
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X, y = check_X_y(X, y)
    assert len(feature_names) == X.shape[1], "feature_names must have the same length as the number of columns in X"
    return X, y, feature_names


def _shrink_tree_rec(dt, shrink_mode, lmb=0,
                     X_train: Optional[Union[pd.DataFrame, npt.NDArray]] = None,
                     X_train_parent: Optional[Union[pd.DataFrame, npt.NDArray]] = None, node=0,
                     parent_node=None, parent_val=None, cum_sum=None):
    """
    Go through the tree and shrink contributions recursively
    Don't call this function directly, use shrink_forest or shrink_tree
    """
    assert shrink_mode in ("hs", "hs_entropy", "hs_entropy_2", 
                           "hs_log_cardinality"),\
                                   "Invalid choice for shrink_mode"
    left = dt.tree_.children_left[node]
    right = dt.tree_.children_right[node]
    feature = dt.tree_.feature[node]
    threshold = dt.tree_.threshold[node]
    parent_num_samples = dt.tree_.n_node_samples[parent_node]
    parent_feature = dt.tree_.feature[parent_node]
    value = deepcopy(dt.tree_.value[node, :, :] / dt.tree_.value[node].sum())

    # cum_sum contains the value of the telescopic sum
    # If root: initialize cum_sum to the value of the root node
    if parent_node is None:
        cum_sum = value
    else:
        # If not root: update cum_sum based on the value of the current node and the parent node
        reg = 1
        if shrink_mode == "hs":
            # Classic hierarchical shrinkage
            reg = 1 + (lmb / parent_num_samples)
        else:
            assert X_train is not None, "Augmented hierarchical shrinkage requires a training dataset"
            assert X_train_parent is not None, "Augmented hierarchical shrinkage requires a training dataset"
            if isinstance(X_train_parent, pd.DataFrame):
                parent_split_feature = X_train_parent.iloc[:, parent_feature]
            else:
                parent_split_feature = pd.Series(X_train_parent[:, parent_feature])

            if shrink_mode == "hs_entropy":
                # Entropy-based shrinkage
                # Note: we can just use the value_counts, scipy.stats.entropy handles normalization
                # i.e. it is not necessary to divide by the total number of samples
                entropy = scipy.stats.entropy(parent_split_feature.value_counts())
                reg = 1 + (lmb * entropy / parent_num_samples)
            elif shrink_mode == "hs_entropy_2":
                # Entropy-based shrinkage, but entropy term is added outside of the fraction
                entropy = scipy.stats.entropy(parent_split_feature.value_counts())
                reg = entropy * (1 + lmb / parent_num_samples)
            elif shrink_mode == "hs_log_cardinality":
                # Cardinality-based shrinkage
                cardinality = len(parent_split_feature.unique())
                reg = 1 + (lmb * np.log(cardinality) / parent_num_samples)
        cum_sum += (value - parent_val) / reg
    
    # Set the value of the node to the value of the telescopic sum
    dt.tree_.value[node, :, :] = cum_sum
    # Update the impurity of the node
    dt.tree_.impurity[node] = 1 - np.sum(np.power(cum_sum, 2))
    # If not leaf: recurse
    if not (left == -1 and right == -1):
        if X_train is not None:
            # If we use augmented HS, we need to filter the training data
            if isinstance(X_train, pd.DataFrame):
                X_train_left = deepcopy(X_train[X_train.iloc[:, feature] <= threshold])
                X_train_right = deepcopy(X_train[X_train.iloc[:, feature] > threshold])
            else:
                X_train_left = deepcopy(X_train[X_train[:, feature] <= threshold])
                X_train_right = deepcopy(X_train[X_train[:, feature] > threshold])
            _shrink_tree_rec(dt, shrink_mode, lmb, X_train_left, X_train, left, node, value, deepcopy(cum_sum))
            _shrink_tree_rec(dt, shrink_mode, lmb, X_train_right, X_train, right, node, value, deepcopy(cum_sum))
        else:
            # For classic HS, no filtering is necessary
            _shrink_tree_rec(dt, shrink_mode, lmb, None, None, left, node, value, deepcopy(cum_sum))
            _shrink_tree_rec(dt, shrink_mode, lmb, None, None, right, node, value, deepcopy(cum_sum))


class ShrinkageWrapper(BaseEstimator):
    def predict(self, X, *args, **kwargs):
        return self.base_estimator.predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        if hasattr(self.base_estimator, 'predict_proba'):
            return self.base_estimator.predict_proba(X, *args, **kwargs)
        else:
            return NotImplemented

    def score(self, X, y, *args, **kwargs):
        if hasattr(self.base_estimator, 'score'):
            return self.base_estimator.score(X, y, *args, **kwargs)
        else:
            return NotImplemented


class AugHSTree(ShrinkageWrapper):
    def __init__(self,
                 base_estimator: BaseEstimator = DecisionTreeClassifier(max_leaf_nodes=20),
                 shrink_mode: str = "hs",
                 lmb: float = 1):
        if shrink_mode not in ["hs", "hs_entropy", "hs_entropy_2", "hs_log_cardinality"]:
            raise ValueError("Invalid choice for shrink_mode")
        self.base_estimator = base_estimator
        self.lmb = lmb
        self.shrink_mode = shrink_mode

    def fit(self, X: Union[npt.NDArray, pd.DataFrame], 
            y: Union[npt.NDArray, pd.DataFrame], **kwargs):
        feature_names = kwargs.pop("feature_names", None)
        X, y, feature_names = _check_fit_arguments(X, y, feature_names=feature_names)
        self.base_estimator = self.base_estimator.fit(X, y, **kwargs)
        _shrink_tree_rec(self.base_estimator, self.shrink_mode, self.lmb, X)
        return self


class AugHSForest(ShrinkageWrapper):
    def __init__(self,
                 base_estimator: BaseEstimator = RandomForestClassifier(n_estimators=100),
                 shrink_mode: str = "hs",
                 lmb: float = 1):
        if shrink_mode not in ["hs", "hs_entropy", "hs_entropy_2", "hs_log_cardinality"]:
            raise ValueError("Invalid choice for shrink_mode")
        self.base_estimator = base_estimator
        self.lmb = lmb
        self.shrink_mode = shrink_mode
    
    def fit(self, X: Union[npt.NDArray, pd.DataFrame], 
            y: Union[npt.NDArray, pd.DataFrame], **kwargs):
        feature_names = kwargs.pop("feature_names", None)
        X, y, feature_names = _check_fit_arguments(X, y, feature_names=feature_names)
        self.base_estimator = self.base_estimator.fit(X, y, **kwargs)
        for dt in self.base_estimator.estimators_:
            _shrink_tree_rec(dt, self.shrink_mode, self.lmb, X)
        return self


class AugHSTreeClassifier(AugHSTree, ClassifierMixin):
    ...

class AugHSForestClassifier(AugHSForest, ClassifierMixin):
    ...

class AugHSTreeRegressor(AugHSTree, RegressorMixin):
    ...

class AugHSForestRegressor(AugHSForest, RegressorMixin):
    ...
