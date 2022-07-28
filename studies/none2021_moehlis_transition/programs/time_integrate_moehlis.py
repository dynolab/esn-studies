#!/usr/bin/env python3
import os
import json
import argparse
import concurrent.futures
import pickle
from functools import partial

import numpy as np

from misc import upload_paths_from_config, generate_random_perturbation
upload_paths_from_config()
from thequickmath.reduced_models.transition_to_turbulence import MoehlisFaisstEckhardtModel, \
    MoehlisFaisstEckhardtPerturbationDynamicsModel
from thequickmath.reduced_models.models import rk4_timestepping


def timeintegrate(ic, delta_t=10**(-3), n_steps=int(1.5*10**6), time_skip=100, stop_condition=lambda s: False):
    print('timeintegrate')
    return rk4_timestepping(m, ic=ic,
                            delta_t=delta_t,
                            n_steps=n_steps, time_skip=time_skip, stop_condition=stop_condition)


def stop_if_close_to_laminar_flow(state, model):
    return np.abs(model.kinetic_energy(state) - model.kinetic_energy(model.laminar_state)) < 0.01


def do_not_stop(state):
    return False


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

    if inputs['equation_type'] == 'full':
        m = MoehlisFaisstEckhardtModel(Re=inputs['re'], L_x=inputs['l_x'] * np.pi, L_z=inputs['l_z'] * np.pi)
    elif inputs['equation_type'] == 'perturbation':
        m = MoehlisFaisstEckhardtPerturbationDynamicsModel(Re=inputs['re'], L_x=inputs['l_x'] * np.pi,
                                                           L_z=inputs['l_z'] * np.pi)
    else:
        raise ValueError(f'Unknown type of equations: {inputs["equation_type"]}')

#    ke_departure = 10**(-4)
#    random_perturbation = generate_random_perturbation(ke_departure)
#    n_ensemble = 20
#    time_until_end = 2000
#    original_ic = training_timeseries[-time_until_end, :]
#    ensemble_ics = [original_ic + generate_random_perturbation(ke_departure) for _ in range(n_ensemble)]
#    ensemble_members = []

    stop_condition = None
    if inputs['stop_condition'] == 'default':
        stop_condition = do_not_stop
    elif inputs['stop_condition'] == 'close_to_laminar_flow':
        stop_condition = partial(stop_if_close_to_laminar_flow, model=m)
    else:
        raise ValueError(f'Unknown stop condition: {inputs["stop_condition"]}')

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.cores) as executor:
        ensemble_members = executor.map(partial(timeintegrate, delta_t=inputs['time_step'],
                                                n_steps=int(inputs['final_time']//inputs['time_step']),
                                                time_skip=int(inputs['save_dt']//inputs['time_step']),
                                                stop_condition=stop_condition),
                                        inputs['initial_conditions'])
    print('Step 1')

    ensemble_members = list(ensemble_members)
    print('Step 2')
    for i, output_filename in enumerate(inputs['output_filenames']):
        print(f'Step {i + 3}')
        with open(output_filename, 'wb') as f:
            n_datapoints = ensemble_members[i].shape[0]
            output_data = {
                'timeseries': ensemble_members[i],
                'time': np.linspace(0, inputs['save_dt'] * n_datapoints, n_datapoints),
            }
            pickle.dump(output_data, f)
    finished_file = open('__finished__', 'w')
    finished_file.close()
