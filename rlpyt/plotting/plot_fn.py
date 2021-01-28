import seaborn as sns
sns.set(

)
import pandas as pd
import os.path as osp
import json
import glob
import csv
import os
import numpy as np

# Common filenames
PROG = 'progress.csv'
PARAMS = 'params.json'
DEBUG = 'debug.log'
VAR_CONFIG = 'variant_config.json'
FILES_PER_RUN = [PROG, PARAMS, DEBUG]
FILES_PER_SETTING = [VAR_CONFIG]

RUN_KEY = 'run_'  # Specify folders across which to average rather than treating as
MERGE_KEY = 'run_ALL'  # For combined results

# Params required for combining std
DEFAULT_TRAJ_WINDOW = 100  # Sample size for

example_dir = '/home/zhef-home/Documents/GitHub/rlpyt/data/local/20210126/160741/example_2a/run_0/'

parent_directory = '/home/david/Documents/GitHub/rlpyt/data/local/HalfCheetahTransfer/All'

def load_single(directory, index_col='Diagnostics/Iteration'):
    """ Load progress.csv, params.json, and debug.log from one directory

    Args:
        directory: Immediate parent directory of experiment files
        index_col: Column to use for indexing the dataframe. Defaults to iteration. Could also use 'Diagnostics/CumSteps'


    Return progress.csv as Pandas dataframe, params as dict. Ignore debug.log for now
    """
    prog_file, params_file, _ = [osp.join(directory, fn) for fn in FILES_PER_RUN]
    return pd.read_csv(prog_file, index_col=index_col), json.load(open(params_file)),

def load_one_csv(progress_csv_path):
    """Load a specific progress.csv file as dict

    Returns dict(col_name: ndarray)
    """
    print("Reading %s" % progress_csv_path)
    entries = dict()
    if progress_csv_path.split('.')[-1] == "csv":
        delimiter = ','
    else:
        delimiter = '\t'
    with open(progress_csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            for k, v in row.items():
                if k not in entries:
                    entries[k] = []
                try:
                    entries[k].append(float(v))
                except:
                    entries[k].append(0.)
    entries = dict([(k, np.array(v)) for k, v in entries.items()])
    return entries

def load_all_csv_and_param(parent_directory):
    """Take above, apply to all runs in directory"""
    runs = [osp.join(parent_directory, d) for d in os.listdir(parent_directory)]  # Run directories
    runs = [d for d in runs if osp.isdir(d)]
    runs_csv = [osp.join(d, PROG) for d in runs]  # Run csvs
    run_params = osp.join(runs[0], PARAMS)  # One params file is fine
    all_entries = [load_one_csv(rcsv) for rcsv in runs_csv]  # List of dicts
    base_entries = all_entries[0]
    for k, v in base_entries.items():
        all_v = [v] + [e[k]for e in all_entries[1:]]  # Get values for all other runs
        base_entries[k] = np.column_stack(all_v)
    all_params = load_one_params(run_params)  # One param dict
    return base_entries, all_params

def combine_entries(entries):
    """Take combined dict object, fix vals"""
    SUFFIXES = ['Average', 'Std', 'Median', 'Min', 'Max']
    SAMPLE_SIZE_KEY = 'Diagnostics/CumCompletedTrajs'  # Used to compute sample size
    SAMPLE_SIZES = np.minimum(entries[SAMPLE_SIZE_KEY], DEFAULT_TRAJ_WINDOW)  #nparray
    TOTAL_SAMPLE_SIZES = SAMPLE_SIZES.sum(-1, keepdims=True)
    keys = list(entries.keys())
    SUFFIX_KEYS = {
        s: [k for k in keys if k.endswith(s)] for s in SUFFIXES
    }  # Keys that have those specific suffixes
    # Weighted average and median by sample size
    for k in SUFFIX_KEYS['Median']:
        entries[k] = np.nansum(entries[k] * SAMPLE_SIZES / TOTAL_SAMPLE_SIZES, axis=-1)
    # Min and max are easy
    for k in SUFFIX_KEYS['Min']:
        entries[k] = np.nanmin(entries[k], axis=-1)
    for k in SUFFIX_KEYS['Max']:
        entries[k] = np.nanmax(entries[k], axis=-1)
    # Standard deviation is fancy. Also do weighted average in here
    for k in SUFFIX_KEYS['Std']:
        # https://www.statstodo.com/CombineMeansSDs_Pgm.php
        avg_k = k[:-3] + 'Average'  # Get mean for calculation
        all_avg = entries[avg_k]  # Mean for each group
        # Compute weighted average here
        entries[avg_k] = np.nansum(entries[k] * SAMPLE_SIZES / TOTAL_SAMPLE_SIZES, axis=-1)
        x = SAMPLE_SIZES * all_avg
        tx = np.nansum(x, axis=-1)
        var = entries[k] ** 2  # Variance is std squared
        xx = var * (SAMPLE_SIZES-1) + (x**2) / SAMPLE_SIZES
        txx = np.nansum(xx, axis=-1)
        tsd = np.sqrt((txx-tx**2/TOTAL_SAMPLE_SIZES.flatten()) / (TOTAL_SAMPLE_SIZES.flatten()-1))
        entries[k] = tsd
    OTHER_KEYS = list(set(keys) - set([v_s for v in SUFFIX_KEYS.values() for v_s in v]))
    for k in OTHER_KEYS: entries[k] = np.nanmean(entries[k], axis=-1)
    return entries

def load_one_params(params_json_path):
    """Load params.json as dict"""
    with open(params_json_path, 'r') as f:
        data = json.loads(f.read())
        if "args_data" in data:
            del data["args_data"]
        if "exp_name" not in data:
            data["exp_name"] = params_json_path.split("/")[-2]
    return data

if __name__ == "__main__":
    example = '/home/david/Documents/GitHub/rlpyt/data/local/20210119/202536/PPOC_Isaac/Ant/run_0/'
    # all_dirs = [d for d in os.walk(parent_directory)]
    direct_parent_directory = '/home/david/Documents/GitHub/rlpyt/data/local/20210119/202536/PPOC_Isaac/Ant'
    entries, params = load_all_csv_and_param('/home/david/Documents/GitHub/rlpyt/data/local/20210119/202536/PPOC_Isaac/Ant')
    entries = combine_entries(entries)
    run_dir = osp.join(direct_parent_directory, MERGE_KEY)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(params, open(osp.join(run_dir, PARAMS), 'w'))  # Dump params
    df = pd.DataFrame(entries)  # Construct df and dump
    df.to_csv(osp.join(run_dir, PROG), index=False)


    print(3)
    # os.walk: (dirpath, dirnames, filenames)
    # There will be a variant_config.json file at the parent directory above runs
    # There is progress.csv, params in each run
    pass