import sys
sys.path.append("../..")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from aughs import ShrinkageClassifier, cross_val_shrinkage
from tqdm import trange
#from argparse import ArgumentParser
import joblib
import os
from datetime import datetime
import time
from matplotlib import pyplot as plt


def simulate_Strobl(n_samples: int, relevance: float):
    X = np.zeros((n_samples, 5))
    X[:, 0] = np.random.normal(0, 1, n_samples)
    n_categories = [2, 4, 10, 20]
    for i in range(1, 5):
        X[:, i] = np.random.choice(
            a=n_categories[i-1], size=n_samples,
            p=np.ones(n_categories[i - 1]) / n_categories[i - 1])
    y = np.zeros(n_samples)
    y[X[:, 1] == 0] = np.random.binomial(
        1, 0.5 - relevance, np.sum(X[:, 1] == 0))
    y[X[:, 1] == 1] = np.random.binomial(
        1, 0.5 + relevance, np.sum(X[:, 1] == 1))
    return X, y

def simulate_Wright_Nembrini(n_samples: int, relevance: float):
    X = np.zeros((n_samples, 5))
    X[:, 0] = np.random.normal(0, 1, n_samples)
    n_categories = [2, 4, 10, 20]
    for i in range(1, 5):
        X[:, i] = np.random.choice(
            a=n_categories[i-1], size=n_samples,
            p=np.ones(n_categories[i - 1]) / n_categories[i - 1])
    y = np.zeros(n_samples)
    y[X[:, 1] == 0] = np.random.binomial(
        1, 0.5 - relevance, np.sum(X[:, 1] == 0))
    y[X[:, 1] == 1] = np.random.binomial(
        1, 0.5 + relevance, np.sum(X[:, 1] == 1))
    return X, y

def run_experiment(lambdas, relevances, shrink_modes, clf_type="rf", 
                   score_fn = "AUC", n_samples=1000, 
                   max_depth=None , verbose=True):
    relevances_str = ["{:.2f}".format(rel)[2:] for rel in relevances]
    result_importances = {rel: {sm: None for sm in shrink_modes}
                          for rel in relevances_str}
    result_scores = {rel: {sm: None for sm in shrink_modes}
                      for rel in relevances_str}
    result_lambdas = {rel: {sm: None for sm in shrink_modes}
                      for rel in relevances_str}
    
    for i, relevance in enumerate(relevances):
        if verbose:
            print("run_experiment, relevance=", relevance)
        rel_str = relevances_str[i]
        X, y = simulate_Strobl(n_samples, relevance)

        # Compute importances for classical RF/DT
        if clf_type == "rf":#n_jobs=5
            clf = RandomForestClassifier(max_depth=max_depth).fit(X, y)
        elif clf_type == "dt":
            clf = DecisionTreeClassifier(max_depth=max_depth).fit(X, y)
        else:
            raise ValueError("Unknown classifier type")
        result_importances[rel_str]["no_shrinkage"] = clf.feature_importances_
        result_lambdas[rel_str]["no_shrinkage"] = 0

        # Compute importances for different HS modes
        if clf_type == "rf":
            hsc = ShrinkageClassifier(RandomForestClassifier(max_depth=max_depth))
        elif clf_type == "dt":
            hsc = ShrinkageClassifier(DecisionTreeClassifier(max_depth=max_depth))
        else:
            raise ValueError("Unknown classifier type")

        for shrink_mode in shrink_modes:#["hs", "hs_entropy", "hs_log_cardinality"]:
            param_grid = {"shrink_mode": [shrink_mode], "lmb": lambdas}
            lmb_scores = cross_val_shrinkage(
                hsc, X, y, param_grid, n_splits=5, score_fn = score_fn, 
                n_jobs=1, return_param_values=False)
            result_scores[rel_str][shrink_mode] = lmb_scores
            best_idx = np.argmax(lmb_scores)
            best_lmb = lambdas[best_idx]
            hsc.set_shrink_params(shrink_mode=shrink_mode, lmb=best_lmb)
            result_importances[rel_str][shrink_mode] = hsc.estimator_.feature_importances_
            result_lambdas[rel_str][shrink_mode] = best_lmb

    return result_importances, result_scores, result_lambdas


def CreateFilePath(fullFilePath, addDate = False):
    fileInfos = os.path.split(os.path.abspath(fullFilePath))
    fname = fileInfos[1]
    out_path = fileInfos[0]

    if addDate:
        out_path = out_path + datetime.now().strftime("-%Y-%m-%d-%H-%M")
    
    try:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    except:
        print("could not create", out_path)

    return out_path+"/", fname
    

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def plot_importances(result, relevance):
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    fig, ax = plt.subplots()
    importances = result[relevance]
    width = 0.1
    for i, key in enumerate(importances.keys()):
        bpl = ax.boxplot(importances[key], positions=np.arange(5) + (i-2)*width,
                         sym='', widths=width, showfliers=False)
        set_box_color(bpl, colors[i])
        ax.plot([], c=colors[i], label=key)

    ax.legend()
    ax.set_title(f"Relevance: {relevance}")
    ax.set_xticks(np.arange(5), ["X1", "X2", "X3", "X4", "X5"])
    return fig, ax

def plot_scores(result, relevance, ylabel="Accuracy"):
    colors = ['blue', 'red', 'green', 'orange']
    fig, ax = plt.subplots()
    scores = result[relevance]
    for i, key in enumerate(scores.keys()):
        # Make line plot averaging over rows
        ax.plot(np.mean(scores[key], axis=0), label=key, c=colors[i])
        # Plot confidence interval
        ci = 1.96 * np.std(scores[key], axis=0) / np.sqrt(scores[key].shape[0])
        ax.fill_between(np.arange(len(scores[key][0])),
                        np.mean(scores[key], axis=0) - ci,
                        np.mean(scores[key], axis=0) + ci,
                        alpha=0.2, color=colors[i])
    
    ax.legend()
    ax.set_title(f"Relevance: {relevance}")
    ax.set_xticks(np.arange(6), [0.1, 1.0, 10.0, 25.0, 50.0, 100.0])
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel(ylabel)
    return fig, ax

def InitDictionary(shrink_modes, relevances_str):
    importances = {
        rel: {
            mode: [] for mode in shrink_modes + ["no_shrinkage"]}
        for rel in relevances_str
    }
    return importances