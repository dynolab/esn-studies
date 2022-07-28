import os
import sys
sys.path.append(os.getcwd())
import time

import numpy as np

from restools.standardised_programs import StandardisedProgramEdge, MoehlisModelIntegrator, EsnIntegrator
from restools.timeintegration import TimeIntegrationLowDimensional
from studies.none2021_moehlis_transition.extensions import LocalPythonTimeIntegrationGraph,\
    RemotePythonTimeIntegrationGraph
from comsdk.communication import LocalCommunication, SshCommunication
from comsdk.research import Research, CreateTaskGraph
from comsdk.graph import Graph, State, Func
from comsdk.edge import Edge, dummy_predicate, make_dump


if __name__ == '__main__':
    local_comm = LocalCommunication.create_from_config()
#    ssh_comm = SshCommunication.create_from_config('atmlxint2')
#    res = Research.open('RC_MOEHLIS', ssh_comm)
    res = Research.open('RC_MOEHLIS')

    ics = []
    source_task = 37
    begin_time = 13940 - 100
    for ic_i in range(1, 101):
        ti = TimeIntegrationLowDimensional(res.get_task_path(source_task), 1)
        ics.append(ti.timeseries[begin_time-10:begin_time].tolist())
    n_ics = len(ics)
    re = 500
    esn_name = 'esn_re_500_trained_wo_lam_event'
    l_x = 1.75
    l_z = 1.2
    n_steps = 2000
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
        'description': f'ESN trajectories for prediction of turbulent-to-laminar transition. '
                       f'We are using new_ts.json dataset (task 37) here starting at T = {begin_time} '
                       f'and predicting for {n_steps} time units.'
    }

#    graph = RemotePythonTimeIntegrationGraph(res, ssh_comm,
#                                             EsnIntegrator(input_filename_key='input_filename', nohup=True),
#                                             input_filename=data['input_filename'],
#                                             task_prefix='ESNPrediction')

    graph = LocalPythonTimeIntegrationGraph(res, local_comm,
                                            EsnIntegrator(input_filename_key='input_filename', nohup=True),
                                            input_filename=data['input_filename'],
                                            task_prefix=f'ESNPredictionOfLaminarization_from_{begin_time}')
    okay = graph.run(data)
    if not okay:
        print(data['__EXCEPTION__'])
    print(data)
