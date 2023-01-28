import os
import sys
sys.path.append(os.getcwd())
import time

import numpy as np

from restools.standardised_programs import StandardisedProgramEdge, MoehlisModelIntegrator, EsnIntegrator
from restools.timeintegration import TimeIntegrationLowDimensional
from studies.none2021_moehlis_transition.extensions import LocalPythonTimeIntegrationGraph,\
    RemotePythonTimeIntegrationGraph, EsnIntegratorWOSynchronization
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
    source_task = 45
    #begin_time = 13940 + 100 + 100 + 100
    for ic_i in range(1, 11):
        ti = TimeIntegrationLowDimensional(res.get_task_path(source_task), ic_i)
        #q = ti.timeseries - np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float64)
        #ti = TimeIntegrationLowDimensional(res.get_task_path(source_task), 1)
        ics.append(ti.timeseries[:10].tolist())
        #ics.append(ti.timeseries[begin_time-10:begin_time].tolist())
    n_ics = len(ics)
    re = 275
    esn_name = f'esn_re_{re}'
    esn_task = 136
    dt_as_defined_by_esn = 5
    #esn_name = 'esn_trained_wo_lam_event'
    l_x = 1.75
    l_z = 1.2
    n_steps = int(100 // dt_as_defined_by_esn)
    data = {
        'res': res,
#        'esn_path': os.path.join(res.local_research_path, esn_name),
        'esn_path': os.path.join(res.get_task_path(esn_task), esn_name),
#        'esn_path': os.path.join(res.remote_research_path, esn_name),
        'dt_as_defined_by_esn': dt_as_defined_by_esn,
        'n_steps': n_steps,
        'final_time': n_steps*dt_as_defined_by_esn,
        're': re,
        'l_x': l_x,
        'l_z': l_z,
        'input_filename': 'inputs.json',
        'output_filename': '1',
        'description': f'ESN prediction without synchronization at Re = {re} for esn trained in task {esn_task}'
    }

#    graph = RemotePythonTimeIntegrationGraph(res, ssh_comm,
#                                             EsnIntegrator(input_filename_key='input_filename', nohup=True),
#                                             input_filename=data['input_filename'],
#                                             task_prefix='ESNPrediction')

    graph = LocalPythonTimeIntegrationGraph(res, local_comm,
                                            EsnIntegratorWOSynchronization(
                                                input_filename_key='input_filename',
                                                nohup=False),
                                            input_filename=data['input_filename'],
                                            task_prefix=f'ESNPredictionWOSync_timestep_{dt_as_defined_by_esn}')
    okay = graph.run(data)
    if not okay:
        print(data['__EXCEPTION__'])
    print(data)
