#!/Users/tony/miniconda3/envs/esn/bin/python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import json
import argparse
import concurrent.futures
import pickle
from functools import partial

import numpy as np

from misc import upload_paths_from_config
upload_paths_from_config()
from pyESN import ESN


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manages different objects related to research.')
    parser.add_argument('--cores', metavar='CORES', type=int, default=4, help='allowed number of cores')
    parser.add_argument('filename', metavar='FILENAME', nargs='?', help='input parameters filename')
    args = parser.parse_args()
    with open(args.filename, 'r') as f:
        inputs = json.load(f)
    os.environ["OMP_NUM_THREADS"] = str(args.cores)
    os.environ["MKL_NUM_THREADS"] = str(args.cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.cores)

    if not 'started' in inputs:
        started_file = open('__started__', 'w')
    else:
        started_file = open(f"i_{inputs['started']}_i", 'w')
    started_file.close()

    with open(inputs['esn_path'], 'rb') as f:
        esn = pickle.load(f)
        esn.enable_noise_while_predicting = True

    prediction = esn.predict(np.ones(inputs['n_steps']))
    with open(inputs['output_filename'], 'wb') as f:
        n_datapoints = prediction.shape[0]
        output_data = {
            'timeseries': prediction,
            'time': np.linspace(0, inputs['dt_as_defined_by_esn'] * inputs['n_steps'], inputs['n_steps']),
        }
        pickle.dump(output_data, f)
    if not 'finished' in inputs:
        finished_file = open('__finished__', 'w')
    else:
        finished_file = open(f"i_{inputs['finished']}_i", 'w')
    finished_file.close()
