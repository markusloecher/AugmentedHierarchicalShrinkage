from StroblSimFuns import *
import argparse

if __name__ == "__main__":
    par_file = "output" + datetime.now().strftime("-%Y-%m-%d-%H-%M") + "/params.pkl"
    input_dir = "output" + datetime.now().strftime("-%Y-%m-%d-%H-%M") 
    output_dir = "plot" + datetime.now().strftime("-%Y-%m-%d-%H-%M")


    parser = argparse.ArgumentParser()
    #parser.add_argument("--pars-file", type=str, default=par_file)
    parser.add_argument("--input-dir", type=str, default=input_dir)
    #parser.add_argument("--output-dir", type=str, default=output_dir)
    parser.add_argument("--scores-ylabel", type=str, default="AUC")
    args = parser.parse_args()
    
    #manual overwrite:
    #fileInfos = os.path.split(os.path.abspath(args.input_file))
    args.output_dir = args.input_dir + "/plots/"
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    result = joblib.load(args.input_dir + "/scores.pkl")
    print("results shape:", result["00"]["hs"].shape)
    for relevance in result.keys():
        fig, ax = plot_scores(result, relevance, args.scores_ylabel)

        fig.savefig(os.path.join(
            args.output_dir, f"scores_{relevance}.png"))