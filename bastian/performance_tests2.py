from imodels.util.data_util import get_clean_dataset
import numpy as np
from beta import ShrinkageClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

clf_datasets = [
    ("heart", "heart", "imodels"),
    ("breast-cancer", "breast_cancer", "imodels"), 
    ("haberman", "haberman", "imodels"), 
    ("ionosphere", "ionosphere", "pmlb"),
    ("diabetes-clf", "diabetes", "pmlb"),
    ("german", "german", "pmlb"),
    ("juvenile", "juvenile_clean", "imodels"),
    ("recidivism", "compas_two_year_clean", "imodels")
]


lmbs = np.arange(0, 50, 1)
for ds_name, id, source in clf_datasets:
    X, y, feature_names = get_clean_dataset(id, data_source=source)
    scores = {}
    print(ds_name)
    #for shrink_mode in ["hs", "hs_entropy", "hs_entropy_2", "hs_log_cardinality"]:
    #    scores[shrink_mode] = []
    #    for lmb in lmbs:
    #        clf = ShrinkageClassifier(shrink_mode=shrink_mode, lmb=lmb)
    #        scores[shrink_mode].append(
    #            cross_val_score(clf, X, y, cv=10, n_jobs=-1,
    #                            scoring="balanced_accuracy").mean())        

    # hs
    shrink_mode="hs"
    scores[shrink_mode] = []
    param_grid = {
    "lmb": [1, 10, 50, 100, 200, 500],
    "shrink_mode": ["hs"]}

    grid_search = GridSearchCV(ShrinkageClassifier(), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    print(best_params)

    for lmb in lmbs:
        clf = ShrinkageClassifier(shrink_mode=shrink_mode, lmb=best_params.get('lmb'))
        #print(clf)
        scores[shrink_mode].append(cross_val_score(clf, X, y, cv=10, n_jobs=-1,
            scoring="balanced_accuracy").mean())    
    
    # hs_entropy
    shrink_mode="hs_entropy"
    scores[shrink_mode] = []
    param_grid = {
    "lmb": [1, 10, 50, 100, 200, 500],
    "shrink_mode": ["hs_entropy"]}

    grid_search = GridSearchCV(ShrinkageClassifier(), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    print(best_params)

    for lmb in lmbs:
        clf = ShrinkageClassifier(shrink_mode=shrink_mode, lmb=best_params.get('lmb'))
        #print(clf)
        scores[shrink_mode].append(cross_val_score(clf, X, y, cv=10, n_jobs=-1,
            scoring="balanced_accuracy").mean())    
    
    # hs_entropy_2
    shrink_mode="hs_entropy_2"
    scores[shrink_mode] = []
    param_grid = {
    "lmb": [1, 10, 50, 100, 200, 500],
    "shrink_mode": ["hs_entropy_2"]}

    grid_search = GridSearchCV(ShrinkageClassifier(), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    print(best_params)

    for lmb in lmbs:
        clf = ShrinkageClassifier(shrink_mode=shrink_mode, lmb=best_params.get('lmb'))
        #print(clf)
        scores[shrink_mode].append(cross_val_score(clf, X, y, cv=10, n_jobs=-1,
            scoring="balanced_accuracy").mean())    
    
    # hs_log_cardinality
    shrink_mode="hs_log_cardinality"
    scores[shrink_mode] = []
    param_grid = {
    "lmb": [1, 10, 50, 100, 200, 500],
    "shrink_mode": ["hs_log_cardinality"]}

    grid_search = GridSearchCV(ShrinkageClassifier(), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    print(best_params)

    for lmb in lmbs:
        clf = ShrinkageClassifier(shrink_mode=shrink_mode, lmb=best_params.get('lmb'))
        #print(clf)
        scores[shrink_mode].append(cross_val_score(clf, X, y, cv=10, n_jobs=-1,
            scoring="balanced_accuracy").mean())    

    # beta
    shrink_mode="beta"
    scores[shrink_mode] = []
    param_grid = {
    "alpha": [1, 10, 50, 100, 200, 500],
    "beta":[1, 10, 50, 100, 200, 500],
    "shrink_mode": ["beta"]}

    grid_search = GridSearchCV(ShrinkageClassifier(), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    print(best_params)
    for lmb in lmbs:
        clf = ShrinkageClassifier(shrink_mode=shrink_mode, alpha=best_params.get('alpha'),beta=best_params.get('beta'))
        #print(clf)
        scores[shrink_mode].append(cross_val_score(clf, X, y, cv=10, n_jobs=-1,
            scoring="balanced_accuracy").mean())    
    
    #print(scores)
    #for key in scores:
    #    #plt.plot(lmbs, scores[key], label=key)
    #    plt.boxplot(scores[key], labels=key)
    
    import numpy as np
    fig, ax = plt.subplots()
    data = list([scores['hs'], scores['hs_entropy'], scores['hs_entropy_2'], scores['hs_log_cardinality'], scores['beta']])
    # basic plot
    ax.boxplot(data, notch=True)

    ax.set_title(ds_name)
    ax.set_xlabel('')
    ax.set_ylabel('Balanced Accuracy')
    xticklabels=['hs', 'hs_entropy', 'hs_entropy_2', 'hs_log_cardinality', 'beta']
    ax.set_xticklabels(xticklabels)
    plt.xticks(fontsize=7)#, rotation=45)
    # add horizontal grid lines
    #ax.yaxis.grid(True)

    plt.savefig(ds_name)
    # show the plot
    #plt.show()


