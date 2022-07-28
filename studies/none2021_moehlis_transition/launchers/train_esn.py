import os
import sys
sys.path.append(os.getcwd())
import time

import numpy as np

from restools.standardised_programs import StandardisedProgramEdge, MoehlisModelIntegrator, EsnIntegrator, EsnTrainer
from restools.timeintegration import TimeIntegrationLowDimensional
from studies.none2021_moehlis_transition.extensions import RemotePythonTimeIntegrationGraph, LocalPythonTimeIntegrationGraph
from comsdk.communication import LocalCommunication, SshCommunication
from comsdk.research import Research, CreateTaskGraph
from comsdk.graph import Graph, State, Func
from comsdk.edge import Edge, dummy_predicate, make_dump


def generate_random_ic_for_lifetime_distr():
    # this algorithm is from Moehlis 2004
    ic = np.random.uniform(-1, 1, size=9)
    k = np.sum(ic**2)
    return np.sqrt(0.3 / k) * ic


if __name__ == '__main__':
    local_comm = LocalCommunication.create_from_config()
#    ssh_comm = SshCommunication.create_from_config('atmlxint2')
#    res = Research.open('RC_MOEHLIS', ssh_comm)
    res = Research.open('RC_MOEHLIS')
    re = 275
    l_x = 1.75
    l_z = 1.2
    data = {
        'res': res,
        'input_filename': 'inputs.json',
        'original_model_parameters': {
            're': re,
            'l_x': l_x,
            'l_z': l_z,
        },
        're': re,
        'final_time': 0,
        'initial_conditions': [],
 #       'training_timeseries_path': os.path.join(res.remote_research_path, f'training_timeseries_re_{re}_new_pert'),
 #       'test_timeseries_path': os.path.join(res.remote_research_path, f'test_timeseries_re_{re}_pert'),
        'training_timeseries_path': os.path.join(res.local_research_path, f'training_timeseries_re_{re}_new_pert.pickle'),
        'test_timeseries_path': os.path.join(res.local_research_path, f'test_timeseries_re_{re}_pert.pickle'),
        'synchronization_len': 10,
        'test_chunk_timeseries_len': 300,
        'spectral_radius_values': [0.5],
        'sparsity_values': [0.1, 0.5, 0.9],
        'reservoir_dimension': 1500,
        'optimal_esn_filename': f'esn_re_{re}',
        'description': f'Training ESN for Re = {re}.',
    }

#    graph = RemotePythonTimeIntegrationGraph(res, local_comm, ssh_comm,
#                                             EsnTrainer(input_filename_key='input_filename', nohup=True),
#                                             input_filename=data['input_filename'],
#                                             output_filenames_key='optimal_esn_filename',
#                                             task_prefix='ESNTrainingPert')

    graph = LocalPythonTimeIntegrationGraph(res, local_comm,
                                            EsnTrainer(input_filename_key='input_filename', nohup=True),
                                            input_filename=data['input_filename'],
                                         #   output_filenames_key='optimal_esn_filename',
                                            task_prefix='ESNTrainingPert')
    
    okay = graph.run(data)
    if not okay:
        print(data['__EXCEPTION__'])
    print(data)
