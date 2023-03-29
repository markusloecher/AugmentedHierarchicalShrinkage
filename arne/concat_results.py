"""
Concatenates the results from different runs of arne/simulation.py into a
single file. Can be used to easily run multiple replications in parallel.
"""
import argparse
import numpy as np
from collections import defaultdict
import joblib
import os
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="arne/output",
                        help="Directory containing the results from the different runs.")
    parser.add_argument("--output-file", type=str, default="results_concat.pkl",
                        help="File to save the concatenated results to.")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.data_dir, "*.pkl"))

    # Signal level -> method -> importances
    result = defaultdict(lambda: defaultdict(list))
    for file in files:
        print(f"Reading {file}")
        data = joblib.load(file)
        for signal_level in data.keys():
            for method in data[signal_level].keys():
                importances = data[signal_level][method]
                result[signal_level][method].append(importances)
    
    for signal_level in result.keys():
        for method in result[signal_level].keys():
            result[signal_level][method] = np.concatenate(result[signal_level][method], axis=0)
            print(f"Signal level {signal_level}, method {method}: {result[signal_level][method].shape}")
        result[signal_level] = dict(result[signal_level])
    result = dict(result)

    joblib.dump(result, args.output_file)