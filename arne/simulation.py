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
    y[X[:, 1] == 0] = np.random.binomial(
        1, 0.5 - relevance, np.sum(X[:, 1] == 0))
    y[X[:, 1] == 1] = np.random.binomial(
        1, 0.5 + relevance, np.sum(X[:, 1] == 1))
    return X, y


def run_experiment(lambdas, relevances, shrink_modes):
    result = {rel: {sm: None for sm in shrink_modes} for rel in relevances}
    all_lmb_scores = {rel: {sm: None for sm in shrink_modes}
                      for rel in relevances}
    for relevance in relevances:
        importances = {}
        X, y = simulate_categorical(1000, relevance)

        # Compute importances for classical RF
        rfc = RandomForestClassifier().fit(X, y)
        importances["random_forest"] = rfc.feature_importances_

        # Compute importances for different HS modes
        hsc = ShrinkageClassifier(RandomForestClassifier())
        for shrink_mode in ["hs", "hs_entropy", "hs_log_cardinality"]:
            lmb_scores = cross_val_lmb(
                hsc, X, y, shrink_mode, lambdas, n_splits=5, n_jobs=1)
            all_lmb_scores[relevance][shrink_mode] = lmb_scores
            best_idx = np.argmax(lmb_scores)
            best_lmb = lambdas[best_idx]
            hsc.set_shrink_params(shrink_mode=shrink_mode, lmb=best_lmb)
            importances[shrink_mode] = hsc.estimator_.feature_importances_
        result[relevance] = importances
    return result, all_lmb_scores


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    from aughs import ShrinkageClassifier, cross_val_lmb
    from tqdm import trange
    from argparse import ArgumentParser
    import joblib

    parser = ArgumentParser()
    parser.add_argument("--n-replications", type=int, default=4)
    parser.add_argument("--lambdas", type=str, default="0,100,10")
    parser.add_argument("--importances-file", type=str,
                        default="simulation.pkl")
    parser.add_argument("--scores-file", type=str,
                        default="simulation_scores.pkl")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    lambdas = np.arange(*[int(x) for x in args.lambdas.split(",")])
    relevances = [0., 0.05, 0.1, 0.15, 0.2]
    shrink_modes = ["hs", "hs_entropy", "hs_log_cardinality"]

    results = joblib.Parallel(n_jobs=args.n_jobs, verbose=10)(
        joblib.delayed(run_experiment)(lambdas, relevances, shrink_modes)
        for _ in range(args.n_replications))
    
    print("DONE")

    # Gather all results
    importances = {
        rel: {
            mode: [] for mode in shrink_modes + ["random_forest"]}
        for rel in relevances
    }

    scores = {
        rel: {
            mode: [] for mode in shrink_modes}
        for rel in relevances
    }

    # Concatenate results
    for result, all_lmb_scores in results:
        for rel in relevances:
            for mode in shrink_modes + ["random_forest"]:
                importances[rel][mode].append(result[rel][mode])
            for mode in shrink_modes:
                scores[rel][mode].append(all_lmb_scores[rel][mode])
    
    # Convert to numpy arrays
    for rel in relevances:
        for mode in shrink_modes + ["random_forest"]:
            importances[rel][mode] = np.array(importances[rel][mode])
        for mode in shrink_modes:
            scores[rel][mode] = np.array(scores[rel][mode])

    # Save to disk
    joblib.dump(importances, args.importances_file)
    joblib.dump(scores, args.scores_file)