from abc import abstractmethod
import pandas as pd
import time
from copy import deepcopy
from typing import Tuple, List

import numpy as np
import scipy
from numpy import typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, balanced_accuracy_score
from joblib import Parallel, delayed


def _check_fit_arguments(X, y, feature_names) -> Tuple[npt.NDArray, npt.NDArray,
                                                       List[str]]:
    if feature_names is None:
        if hasattr(X, "columns"):
            feature_names = X.columns
        else:
            X, y = check_X_y(X, y)
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    else:
        assert len(feature_names) == X.shape[1],\
            "Number of feature names must match number of features"
    X, y = check_X_y(X, y)
    return X, y, feature_names


def _normalize_value(dt, node):
    if isinstance(dt, RegressorMixin):
        return dt.tree_.value[node, :, :]
    # Normalize to probability vector
    return dt.tree_.value[node, :, :] / dt.tree_.n_node_samples[node]


def _update_tree_values(dt, value, node):
    # Set the value of the node to the value of the telescopic sum
    # assert not np.isnan(value).any(), "Cumulative sum is NaN"
    dt.tree_.value[node, :, :] = value
    # Update the impurity of the node
    dt.tree_.impurity[node] = 1 - np.sum(np.power(value, 2))
    # assert not np.isnan(dt.tree_.impurity[node]), "Impurity is NaN"


class ShrinkageEstimator(BaseEstimator):
    def __init__(self, base_estimator: BaseEstimator = None,
                 shrink_mode: str = "hs", lmb: float = 1,
                 random_state=None):
        self.base_estimator = base_estimator
        self.shrink_mode = shrink_mode
        self.lmb = lmb
        self.random_state = random_state
    
    @abstractmethod
    def get_default_estimator(self):
        raise NotImplemented
    
    def _compute_node_entropies(self, dt, X_train, node=0, entropies=None, cardinalities=None):
        left = dt.tree_.children_left[node]
        right = dt.tree_.children_right[node]
        feature = dt.tree_.feature[node]

        if entropies is None:
            entropies = np.zeros(len(dt.tree_.n_node_samples))
            cardinalities = np.zeros(len(dt.tree_.n_node_samples))

        # If not leaf node, compute entropy and cardinality of the node
        if not (left == -1 and right == -1):
            split_feature = X_train[:, feature]
            _, counts = np.unique(split_feature, return_counts=True)
            entropies[node] = scipy.stats.entropy(counts)
            cardinalities[node] = len(counts)

            # Recursively compute entropy and cardinality of the children
            # TODO split train data into left and right
            self._compute_node_entropies(dt, X_train, left, entropies, cardinalities)
            self._compute_node_entropies(dt, X_train, right, entropies, cardinalities)
        return entropies, cardinalities

    
    def _shrink_tree_rec(self, dt, dt_idx, node=0, parent_node=None,
                         parent_val=None, cum_sum=None):
        """
        Go through the tree and shrink contributions recursively
        """
        left = dt.tree_.children_left[node]
        right = dt.tree_.children_right[node]
        parent_num_samples = dt.tree_.n_node_samples[parent_node]
        value = _normalize_value(dt, node)

        # cum_sum contains the value of the telescopic sum
        # If root: initialize cum_sum to the value of the root node
        if parent_node is None:
            cum_sum = value
        else:
            # If not root: update cum_sum based on the value of the current node and the parent node
            reg = 1
            if self.shrink_mode == "hs":
                # Classic hierarchical shrinkage
                reg = 1 + (self.lmb / parent_num_samples)
            else:
                if self.shrink_mode == "hs_entropy":
                    # Entropy-based shrinkage
                    entropy = self.entropies_[dt_idx][parent_node]
                    reg = 1 + (self.lmb * entropy / parent_num_samples)
                elif self.shrink_mode == "hs_log_cardinality":
                    # Cardinality-based shrinkage
                    cardinality = self.cardinalities_[dt_idx][parent_node]
                    reg = 1 + (self.lmb * np.log(cardinality) / parent_num_samples)
            cum_sum += (value - parent_val) / reg
        _update_tree_values(dt, cum_sum, node)
        # If not leaf: recurse
        if not (left == -1 and right == -1):
            self._shrink_tree_rec(dt, dt_idx, left, node, value, cum_sum.copy())
            self._shrink_tree_rec(dt, dt_idx, right, node, value, cum_sum.copy())

    def fit(self, X, y, **kwargs):
        X, y = self._validate_arguments(X, y, kwargs.pop("feature_names", None))

        if self.base_estimator is not None:    
            self.estimator_ = clone(self.base_estimator)
        else:
            self.estimator_ = self.get_default_estimator()

        self.estimator_.set_params(random_state=self.random_state)
        self.estimator_.fit(X, y, **kwargs)

        # Save a copy of the original estimator
        self.orig_estimator_ = deepcopy(self.estimator_)

        # Compute node entropies
        self.entropies_ = []
        self.cardinalities_ = []
        if hasattr(self.estimator_, "estimators_"):  # Random Forest
            for estimator in self.estimator_.estimators_:
                entropies, cardinalities = self._compute_node_entropies(estimator, X)
                self.entropies_.append(entropies)
                self.cardinalities_.append(cardinalities)
        else:  # Single tree
            entropies, cardinalities = self._compute_node_entropies(self.estimator_, X)
            self.entropies_.append(entropies)
            self.cardinalities_.append(cardinalities)

        # Apply hierarchical shrinkage
        self.shrink()
        return self
    
    def shrink(self):
        if hasattr(self.estimator_, "estimators_"):  # Random Forest
            for i, estimator in enumerate(self.estimator_.estimators_):
                self._shrink_tree_rec(estimator, i)
        else:  # Single tree
            self._shrink_tree_rec(self.estimator_, 0)
    
    def set_shrink_params(self, X, shrink_mode=None, lmb=None):
        if shrink_mode is not None:
            self.shrink_mode = shrink_mode
        if lmb is not None:
            self.lmb = lmb
        if not hasattr(self, "entropies_"):
            raise ValueError("Cannot set shrinkage parameters before fitting")
        
        # Reset the estimator to the original one
        self.estimator_ = deepcopy(self.orig_estimator_)
        
        # Apply hierarchical shrinkage
        self.shrink()

    def _validate_arguments(self, X, y, feature_names):
        if self.shrink_mode not in ["hs", "hs_entropy",
                                    "hs_log_cardinality"]:
            raise ValueError("Invalid choice for shrink_mode")
        X, y, feature_names = _check_fit_arguments(
            X, y, feature_names=feature_names)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = feature_names
        return X, y

    def predict(self, X, *args, **kwargs):
        check_is_fitted(self)
        return self.estimator_.predict(X, *args, **kwargs)

    def score(self, X, y, *args, **kwargs):
        check_is_fitted(self)
        return self.estimator_.score(X, y, *args, **kwargs)


class ShrinkageClassifier(ShrinkageEstimator, ClassifierMixin):
    def get_default_estimator(self):
        return DecisionTreeClassifier()

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        self.classes_ = self.estimator_.classes_
        return self
    
    def predict_proba(self, X, *args, **kwargs):
        check_is_fitted(self)
        return self.estimator_.predict_proba(X, *args, **kwargs)


class ShrinkageRegressor(ShrinkageEstimator, RegressorMixin):
    def get_default_estimator(self):
        return DecisionTreeRegressor()


def cross_val_lmb(shrinkage_estimator, X, y, shrink_mode, lmb_range, n_splits,
                  score_fn="balanced_accuracy", n_jobs=-1):
    lmb_scores = []
    cv = KFold(n_splits=n_splits, shuffle=True)

    def _single_fold(train_index, test_index, X, y, lmbs, shrink_mode):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        shrinkage_estimator.fit(X_train, y_train)

        scores = []
        for lmb in lmbs:
            shrinkage_estimator.set_shrink_params(X_train, shrink_mode=shrink_mode, lmb=lmb)
            if score_fn == "balanced_accuracy":
                scores.append(balanced_accuracy_score(y_test, shrinkage_estimator.predict(X_test)))
            elif score_fn == "mse":
                scores.append(mean_squared_error(y_test, shrinkage_estimator.predict(X_test)))
            else:
                raise ValueError("Invalid score function")
        return scores

    if n_jobs != 1:
        lmb_scores = np.array(Parallel(n_jobs=-1)(delayed(_single_fold)(
            train_index, test_index, X, y, lmb_range, shrink_mode)
            for train_index, test_index in cv.split(X)))
    else:
        lmb_scores = np.array([_single_fold(
            train_index, test_index, X, y, lmb_range, shrink_mode)
            for train_index, test_index in cv.split(X)])

    return np.average(lmb_scores, axis=0)



if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    check_estimator(ShrinkageClassifier(RandomForestClassifier()))
    check_estimator(ShrinkageRegressor(RandomForestRegressor()))