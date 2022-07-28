import os
import sys
sys.path.append(os.getcwd())
import time
import json

import numpy as np

from restools.standardised_programs import StandardisedProgramEdge, MoehlisModelIntegrator, EsnIntegrator
from restools.timeintegration import TimeIntegrationLowDimensional
from studies.none2021_predicting_transition_using_reservoir_computing.extensions import LocalPythonTimeIntegrationGraph,\
    RemotePythonTimeIntegrationGraph
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

    ics = []
    source_task = 83
    with open(os.path.join(res.get_task_path(source_task), 'inputs.json'), 'r') as f:
        inputs = json.load(f)
    n_ics = np.max(np.array(inputs['trajectory_numbers'], dtype=int))
    for ic_i in range(1, n_ics + 1):
        ti = TimeIntegrationLowDimensional(res.get_task_path(source_task), ic_i)
        #q = ti.timeseries - np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float64)
        #ti = TimeIntegrationLowDimensional(res.get_task_path(source_task), 1)
        ics.append(ti.timeseries[:10].tolist())
        #ics.append(ti.timeseries[begin_time-10:begin_time].tolist())
    n_ics = len(ics)
    re = inputs['re']
    esn_name = f'esn_re_{re}_trained_wo_lam_event'
    #esn_name = 'esn_trained_wo_lam_event'
    l_x = inputs['l_x']
    l_z = inputs['l_z']
    n_steps = 300
    data = {
        'res': res,
        'esn_path': os.path.join(res.local_research_path, esn_name),
#        'esn_path': os.path.join(res.get_task_path(80), esn_name),
#        'esn_path': os.path.join(res.remote_research_path, esn_name),
        'dt_as_defined_by_esn': 1,
        'n_steps': n_steps,
        'final_time': n_steps,
        're': re,
        'l_x': l_x,
        'l_z': l_z,
        'initial_conditions': ics,
        'input_filename': 'inputs.json',
        'output_filenames': [str(i) for i in range(1, n_ics + 1)],
        'trajectory_numbers': inputs['trajectory_numbers'],
        'energy_levels': inputs['energy_levels'],
        'rps': inputs['rps'],
        'description': f'ESN trajectories for laminarization probability analysis at Re = {re}. '
                       f'Initial conditions are taken from task {source_task}. '
                       f'Noise is disabled while predicting'
    }

#    graph = RemotePythonTimeIntegrationGraph(res, ssh_comm,
#                                             EsnIntegrator(input_filename_key='input_filename', nohup=True),
#                                             input_filename=data['input_filename'],
#                                             task_prefix='ESNPrediction')

    graph = LocalPythonTimeIntegrationGraph(res, local_comm,
                                            EsnIntegrator(input_filename_key='input_filename', nohup=True),
                                            input_filename=data['input_filename'],
                                            task_prefix=f'ESNLaminarizationProbabilityStudyNoNoise')
    okay = graph.run(data)
    if not okay:
        print(data['__EXCEPTION__'])
    print(data)
