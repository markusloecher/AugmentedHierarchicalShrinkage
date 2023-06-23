from StroblSimFuns import *
import argparse
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n-replications", type=int, default=8)
    parser.add_argument("--importances-file", type=str,
                        default="output/importances.pkl")
    parser.add_argument("--clf-type", type=str, default="rf")
    parser.add_argument("--scores-file", type=str,
                        default="output/scores.pkl")
    parser.add_argument("--lambdas-file", type=str,
                        default="output/bestlambdas.pkl")
    parser.add_argument("--pars-file", type=str,
                        default="output/params.pkl")
    parser.add_argument("--score-fn", type=str,
                        default="AUC")
    parser.add_argument("--test-run", type=str,
                        default="no")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--scores-ylabel", type=str, default="AUC")
    parser.add_argument("--plot-dir", type=str, default="plots")
    args = parser.parse_args()

    lambdas = [1.0, 5, 10.0, 25.0, 50.0, 100.0, 150, 200]
    relevances = [0., 0.1, 0.2]#[0., 0.05, 0.1, 0.15, 0.2]
    shrink_modes = ["hs", "hs_entropy", "hs_log_cardinality", "hs_global_entropy"]

    if args.test_run == "yes":
        print("running a quick test")
        lambdas = [0.1, 1.0]
        relevances = [ 0.1]
        shrink_modes = ["hs_global_entropy"]
        args.n_replications = 1
        args.clf_type="dt"
        args.n_samples=100
        args.plot_dir = None

    relevances_str = ["{:.2f}".format(rel)[2:] for rel in relevances]

    start = time.time()

    results = joblib.Parallel(n_jobs=args.n_jobs, verbose=10)(
        joblib.delayed(run_experiment)(lambdas, relevances, shrink_modes,
                                       args.clf_type, args.score_fn, args.n_samples,
                                       args.max_depth)
        for _ in range(args.n_replications))
    end = time.time()
    print("run_experiment took:", end - start)
    # Gather all results
   
    importances = InitDictionary(shrink_modes+ ["no_shrinkage"], relevances_str)
    scores = InitDictionary(shrink_modes, relevances_str)
    best_lambdas = InitDictionary(shrink_modes, relevances_str)

    # Concatenate results
    for result_importances, result_scores, result_lambdas in results:
        for rel in relevances_str:
            for mode in shrink_modes + ["no_shrinkage"]:
                importances[rel][mode].append(result_importances[rel][mode])
            for mode in shrink_modes:
                scores[rel][mode].append(result_scores[rel][mode])
                best_lambdas[rel][mode].append(result_lambdas[rel][mode])
    
    # Convert to numpy arrays
    for rel in relevances_str:
        for mode in shrink_modes + ["no_shrinkage"]:
            importances[rel][mode] = np.array(importances[rel][mode])
        for mode in shrink_modes:
            scores[rel][mode] = np.array(scores[rel][mode])
            best_lambdas[rel][mode] = np.array(best_lambdas[rel][mode])

    #scores_with_type = {}
    #scores_with_type[args.score_fn] = scores
    pars = {"n_samples" : args.n_samples,
            "n_replications" : args.n_replications,
            "lambdas" : lambdas,
            "clf_type" : args.clf_type,
            "score_fn" : args.score_fn,
            "max_depth": args.max_depth}
     
    # Save to disk
    out_path_imp, fname_imp = CreateFilePath(args.importances_file, addDate =True)
    out_path_scores,fname_scores = CreateFilePath(args.scores_file, addDate =True)
    out_path_lambdas,fname_lambdas = CreateFilePath(args.lambdas_file, addDate =True)
    out_path_pars, fname_pars = CreateFilePath(args.pars_file, addDate =True)

    joblib.dump(importances, out_path_imp+fname_imp)
    joblib.dump(scores, out_path_scores+fname_scores)
    joblib.dump(best_lambdas, out_path_lambdas+fname_lambdas)
    joblib.dump(pars, out_path_pars+fname_pars)


if args.plot_dir != None:
    #scores first
    output_dir_scores = out_path_scores + args.plot_dir
    if not os.path.isdir(output_dir_scores):
        os.makedirs(output_dir_scores)
    
    result = scores
    print("scores shape:", result[relevances_str[0]][shrink_modes[0]].shape)
    for relevance in result.keys():
        fig, ax = plot_scores(result, relevance, lambdas, args.scores_ylabel)

        fig.savefig(os.path.join(
            output_dir_scores, f"scores_{relevance}.png"))
        
    #importances next
    output_dir_imp = out_path_imp + args.plot_dir
    if not os.path.isdir(output_dir_imp):
        os.makedirs(output_dir_imp)
    
    result = importances
    print("importances shape:", result[relevances_str[0]][shrink_modes[0]].shape)
    for relevance in result.keys():
        fig, ax = plot_importances(result, relevance)

        fig.savefig(os.path.join(
            output_dir_imp, f"importances_{relevance}.png"))