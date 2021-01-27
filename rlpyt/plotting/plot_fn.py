import seaborn as sns
sns.set(

)
import pandas as pd
import os.path as osp
import json
import glob

# Common filenames
PROG = 'progress.csv'
PARAMS = 'params.json'
DEBUG = 'debug.log'
FILES = [PROG, PARAMS, DEBUG]


example_dir = '/home/zhef-home/Documents/GitHub/rlpyt/data/local/20210126/160741/example_2a/run_0/'

def load_single(directory, index_col='Diagnostics/Iteration'):
    """ Load progress.csv, params.json, and debug.log from one directory

    Args:
        directory: Immediate parent directory of experiment files
        index_col: Column to use for indexing the dataframe. Defaults to iteration. Could also use 'Diagnostics/CumSteps'


    Return progress.csv as Pandas dataframe, params as dict. Ignore debug.log for now
    """
    prog_file, params_file, _ = [osp.join(directory, fn) for fn in FILES]
    return pd.read_csv(prog_file, index_col=index_col), json.load(open(params_file)),

def load_all(directory):
    """ Load progress.csv and params.json from all directories beneath specified directory

    """
    pass

if __name__ == "__main__":
    df, params = load_single(example_dir)
    print(3)
    pass