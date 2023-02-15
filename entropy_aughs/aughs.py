from copy import deepcopy
import scipy
import numpy as np


def shrink_forest(rf, shrink_mode, lmb=0, X_train=None):
    assert shrink_mode in ("hs", "hs_entropy", "hs_entropy_2", "hs_log_cardinality"), "Invalid value for shrink_mode"
    if shrink_mode in ("hs_entropy", "hs_cardinality"):
        assert X_train is not None, "Entropy/cardinality shrinkage requires training data"
    
    rf_copy = deepcopy(rf)
    for tree in rf_copy.estimators_:
        _shrink_tree_rec(tree, shrink_mode, lmb, X_train)
    return rf_copy


def shrink_tree(dt, shrink_mode, lmb=0, X_train=None):
    assert shrink_mode in ("hs", "hs_entropy", "hs_entropy_2", "hs_log_cardinality"), "Invalid value for shrink_mode"
    if shrink_mode in ("hs_entropy", "hs_cardinality"):
        assert X_train is not None, "Entropy/cardinality shrinkage requires training data"
    
    result = deepcopy(dt)
    _shrink_tree_rec(result, shrink_mode, lmb, X_train)
    return result


def _shrink_tree_rec(dt, shrink_mode, lmb=0,
                     X_train=None, X_train_parent=None, node=0,
                     parent_node=None, parent_val=None, cum_sum=None):
    """
    Go through the tree and shrink contributions recursively
    Don't call this function directly, use shrink_forest or shrink_tree
    """
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
        if shrink_mode == "hs":
            # Classic hierarchical shrinkage
            reg = 1 + (lmb / parent_num_samples)
        elif shrink_mode == "hs_entropy":
            # Entropy-based shrinkage
            # Note: we can just use the value_counts, scipy.stats.entropy handles normalization
            # i.e. it is not necessary to divide by the total number of samples
            entropy = scipy.stats.entropy(X_train_parent.iloc[:, parent_feature].value_counts())
            reg = 1 + (lmb * entropy / parent_num_samples)
        elif shrink_mode == "hs_entropy_2":
            # Entropy-based shrinkage, but entropy term is added outside of the fraction
            entropy = scipy.stats.entropy(X_train_parent.iloc[:, parent_feature].value_counts())
            reg = entropy * (1 + lmb / parent_num_samples)
        elif shrink_mode == "hs_log_cardinality":
            # Cardinality-based shrinkage
            cardinality = len(X_train_parent.iloc[:, parent_feature].unique())
            reg = 1 + (lmb * np.log(cardinality) / parent_num_samples)
        cum_sum += (value - parent_val) / reg
    
    # Set the value of the node to the value of the telescopic sum
    dt.tree_.value[node, :, :] = cum_sum
    # Update the impurity of the node
    dt.tree_.impurity[node] = 1 - np.sum(np.power(cum_sum, 2))
    # If not leaf: recurse
    if not (left == -1 and right == -1):
        # TODO this is where we need to filter X_train
        X_train_left = deepcopy(X_train[X_train.iloc[:, feature] <= threshold])
        X_train_right = deepcopy(X_train[X_train.iloc[:, feature] > threshold])
        _shrink_tree_rec(dt, shrink_mode, lmb, X_train_left, X_train, left, node, value, deepcopy(cum_sum))
        _shrink_tree_rec(dt, shrink_mode, lmb, X_train_right, X_train, right, node, value, deepcopy(cum_sum))