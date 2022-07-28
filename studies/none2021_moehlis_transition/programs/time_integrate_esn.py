#!/usr/bin/env python3
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


def timeintegrate(ics, n_steps, esn):
    ics_np = np.array(ics)
    esn.synchronize(np.ones(len(ics)), ics_np)
    return np.r_[ics_np, esn.predict(np.ones(n_steps - len(ics)))]


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

#    with concurrent.futures.ProcessPoolExecutor(max_workers=args.cores) as executor:
#        ensemble_members = executor.map(partial(timeintegrate, n_steps=inputs['n_steps'], esn=esn),
#                                        inputs['initial_conditions'])

    ensemble_members = []
    for ics in inputs['initial_conditions']:
        ensemble_members.append(timeintegrate(ics, n_steps=inputs['n_steps'], esn=esn))

    #ensemble_members = list(ensemble_members)
    for i, output_filename in enumerate(inputs['output_filenames']):
        with open(output_filename, 'wb') as f:
            n_datapoints = ensemble_members[i].shape[0]
            output_data = {
                'timeseries': ensemble_members[i],
                'time': np.linspace(0, inputs['dt_as_defined_by_esn'] * inputs['n_steps'], inputs['n_steps']),
            }
            pickle.dump(output_data, f)
    if not 'finished' in inputs:
        finished_file = open('__finished__', 'w')
    else:
        finished_file = open(f"i_{inputs['finished']}_i", 'w')
    finished_file.close()
