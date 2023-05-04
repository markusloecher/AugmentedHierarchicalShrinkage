from StroblSimFuns import *
import argparse
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=str,
                        default="output/")
    parser.add_argument("--shap-file", type=str,
                        default="shap_vals.pkl")
    parser.add_argument("--lambdas-file", type=str,
                        default="bestlambdas.pkl")
    parser.add_argument("--test-run", type=str,
                        default="no")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--scores-ylabel", type=str, default="AUC")
    parser.add_argument("--plot-dir", type=str, default="plots")
    args = parser.parse_args()

    scores = joblib.load(args.input_dir + "/scores.pkl")
    pars = joblib.load(args.input_dir + "/params.pkl")
    best_lambdas = joblib.load(args.input_dir + "/bestlambdas.pkl")

    lambdas = pars["lambdas"]
    if "max_depth" in pars.keys():
        max_depth = pars["max_depth"]
    else:
        max_depth=None
    clf_type = pars["clf_type"]
    n_samples = pars["n_samples"]
    relevances_str = list(scores.keys())
    shrink_modes = list(scores[relevances_str[0]].keys())
    # if "no_shrinkage" not in shrink_modes:
    #     shrink_modes = shrink_modes + ["no_shrinkage"]
    #     for i in relevances_str:
    #         best_lambdas[i]["no_shrinkage"] = 0
    relevances = [float(i)/100 for i in relevances_str]
    d_scores = scores[relevances_str[0]][shrink_modes[0]].shape
    n_replications = d_scores[0]

    start = time.time()

    if args.test_run == "yes":
        print("running a quick test")
        relevances = [0., 0.2]
        relevances_str = ["{:.2f}".format(rel)[2:] for rel in relevances]
        verbose = 2
        n_samples = 50
        shrink_modes = ["hs", "hs_entropy"]
        results = compute_shap(best_lambdas, relevances, shrink_modes,
                clf_type, n_samples,max_depth, verbose=3)
        print("results:", results)
        joblib.dump(results, args.input_dir + "/shap_temp.pkl")
        results = [results]
        #sys.exit(0)
    else:    
        relevances_str = ["{:.2f}".format(rel)[2:] for rel in relevances]

        results = joblib.Parallel(n_jobs=args.n_jobs, verbose=10)(
            joblib.delayed(compute_shap)(best_lambdas, relevances, shrink_modes,
                                       clf_type, n_samples,
                                       max_depth)
            for _ in range(n_replications))
    
    end = time.time()
    print("run_shap took:", end - start)
    # Gather all results
    shrink_modes = shrink_modes + ["no_shrinkage"]
    shap_vals = InitDictionary(shrink_modes, relevances_str )
    # Concatenate results
    for result_shap in results:
        for rel in relevances_str:
            for mode in shrink_modes :
                shap_vals[rel][mode].append(result_shap[rel][mode])
            
    # Convert to numpy arrays
    for rel in relevances_str:
        for mode in shrink_modes :
            shap_vals[rel][mode] = np.array(shap_vals[rel][mode])
        
 
    # Save to disk
    joblib.dump(shap_vals, args.input_dir + "/" + args.shap_file)
    


