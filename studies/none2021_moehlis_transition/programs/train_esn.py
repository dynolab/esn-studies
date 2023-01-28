#!/Users/tony/miniconda3/envs/esn/bin/python
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import json
import argparse
import concurrent.futures
import pickle
from functools import partial

import numpy as np

from misc import upload_paths_from_config
upload_paths_from_config()
from pyESN import optimal_esn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manages different objects related to research.')
    parser.add_argument('--cores', metavar='CORES', type=int, default=4, help='allowed number of cores')
    parser.add_argument('filename', metavar='FILENAME', nargs='?', help='input parameters filename')
    args = parser.parse_args()
    with open(args.filename, 'r') as f:
        inputs = json.load(f)

    if not 'started' in inputs:
        started_file = open('__started__', 'w')
    else:
        started_file = open(inputs['started'], 'w')
    started_file.close()

    os.environ["OMP_NUM_THREADS"] = str(args.cores)
    os.environ["MKL_NUM_THREADS"] = str(args.cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.cores)

    with open(inputs['training_timeseries_path'], 'rb') as f:
        d = pickle.load(f)
        training_timeseries = d['timeseries']
    with open(inputs['test_timeseries_path'], 'rb') as f:
        d = pickle.load(f)
        test_timeseries = d['timeseries']
    test_chunk_len = inputs['test_chunk_timeseries_len'] + inputs['synchronization_len']
    n_chunks = int(len(test_timeseries) // test_chunk_len)
    test_timeseries_set = [test_timeseries[test_chunk_len * i: test_chunk_len * (i+1)] for i in range(n_chunks)]
    esn = optimal_esn(training_timeseries, test_timeseries_set,
                      spectral_radius_values=inputs['spectral_radius_values'],
                      sparsity_values=inputs['sparsity_values'],
                      n_reservoir=inputs['reservoir_dimension'],
                      trial_number=inputs['trial_number'],
                      random_seed_starts_at=inputs['random_seed_starts_at'])
    print(inputs['optimal_esn_filename'])
    with open(inputs['optimal_esn_filename'], 'wb') as f:
        pickle.dump(esn, f)
    if not 'finished' in inputs:
        finished_file = open('__finished__', 'w')
    else:
        finished_file = open(inputs['finished'], 'w')
    finished_file.close()
