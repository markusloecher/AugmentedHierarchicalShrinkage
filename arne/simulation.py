import numpy as np


def simulate_categorical(n_samples: int, relevance: float):
    X = np.zeros((n_samples, 5))
    X[:, 0] = np.random.normal(0, 1, n_samples)
    n_categories = [2, 4, 10, 20]
    for i in range(1, 5):
        X[:, i] = np.random.choice(
            a=n_categories[i-1], size=n_samples,
              p=np.ones(n_categories[i - 1]) / n_categories[i - 1])
    y = np.zeros(n_samples)
    y[X[:, 1] == 0] = np.random.binomial(1, 0.5 - relevance, np.sum(X[:, 1] == 0))
    y[X[:, 1] == 1] = np.random.binomial(1, 0.5 + relevance, np.sum(X[:, 1] == 1))
    return X, y


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    from aughs import ShrinkageClassifier, cross_val_lmb
    from tqdm import tqdm
    from argparse import ArgumentParser
    import joblib

    parser = ArgumentParser()
    parser.add_argument("--n-replications", type=int, default=1)
    parser.add_argument("--lambdas", type=str, default="0,100,1")
    parser.add_argument("--output-file", type=str, default="simulation.pkl")
    args = parser.parse_args()

    N_REPLICATIONS = args.n_replications
    LAMBDAS = np.arange(*[int(x) for x in args.lambdas.split(",")])
    
    result = {}
    prog_relevance = tqdm([0., 0.05, 0.1, 0.15, 0.2])
    for relevance in prog_relevance:
        prog_relevance.set_description(f"Relevance: {relevance}")
        importances = {
            key: np.zeros((N_REPLICATIONS, 5))
            for key in ["Random Forest", "Hierarchical Shrinkage", "HS: Entropy",
                        "HS: Entropy (2)", "HS: log cardinality"]
        }
        prog_replication = tqdm(range(N_REPLICATIONS), desc="Replication")
        for i in prog_replication:
            X, y = simulate_categorical(1000, relevance)
            
            # Compute importances for classical RF
            rfc = RandomForestClassifier(n_estimators=5).fit(X, y)
            importances["Random Forest"][i, :] = rfc.feature_importances_

            # Compute importances for different HS modes
            hsc = ShrinkageClassifier(RandomForestClassifier(n_estimators=5))
            for shrink_mode, key in zip(["hs", "hs_entropy", "hs_entropy_2", "hs_log_cardinality"],
                                        ["Hierarchical Shrinkage", "HS: Entropy", "HS: Entropy (2)", "HS: log cardinality"]):
                lmb_scores = cross_val_lmb(hsc, X, y, shrink_mode, LAMBDAS, n_splits=5, n_jobs=-1)
                best_idx = np.argmax(lmb_scores)
                best_lmb = LAMBDAS[best_idx]
                hsc.lmb = best_lmb
                hsc.shrink_mode = shrink_mode
                hsc.fit(X, y)
                importances[key][i, :] = hsc.estimator_.feature_importances_
        result[relevance] = importances
    joblib.dump(result, args.output_file)