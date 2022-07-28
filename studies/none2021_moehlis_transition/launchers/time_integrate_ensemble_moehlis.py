import os
import sys
sys.path.append(os.getcwd())
import time

import numpy as np

from restools.standardised_programs import StandardisedIntegrator, StandardisedProgramEdge, MoehlisModelIntegrator
from studies.none2021_moehlis_transition.extensions import LocalPythonTimeIntegrationGraph,\
    RemotePythonTimeIntegrationGraph, generate_random_ic_for_lifetime_distr
from comsdk.communication import LocalCommunication, SshCommunication
from comsdk.research import Research, CreateTaskGraph
from comsdk.graph import Graph, State, Func
from comsdk.edge import Edge, dummy_predicate, dummy_edge, dummy_morphism, InOutMapping, DownloadFromRemoteEdge, \
    UploadOnRemoteEdge, make_dump, make_composite_func


#class MoehlisModelGraph(Graph):
#    def __init__(self, res, comm, input_file_key, input_filename, relative_keys=(), keys_mapping={}, task_prefix=''):
#        def task_name_maker(d):
#            task_name = task_prefix
#            task_name += '_R_{}'.format(d['re'])
#            task_name += '_T_{}'.format(d['final_time'])
#            task_name += '_ens_{}'.format(len(d['initial_conditions']))
#            return task_name
#        task_start, task_end = CreateTaskGraph.create_branch(res, task_name_maker=task_name_maker)
#        ti_init, ti_term = MoehlisModelGraph.create_branch(comm, input_file_key, relative_keys=relative_keys,
#                                                           keys_mapping=keys_mapping)
#        dumping_edge = Edge(dummy_predicate, Func())
#        task_end.connect_to(ti_init, edge=dumping_edge)
#        dumping_edge.postprocess = make_dump(input_filename, omit=['res'], method='json')
#        super().__init__(task_start, ti_term)
#
#    @staticmethod
#    def create_branch(comm, input_file_key, relative_keys=(), keys_mapping={}, array_keys_mapping=None):
#        s_init = State('READY_FOR_MOEHLIS', array_keys_mapping=array_keys_mapping)
#        s_term = State('MOEHLIS_FINISHED')
#        integrate_edge = StandardisedProgramEdge(MoehlisModelIntegrator(input_filename_key=input_file_key),
#                                                 comm, relative_keys=relative_keys, keys_mapping=keys_mapping)
#        s_init.connect_to(s_term, edge=integrate_edge)
#        return s_init, s_term


if __name__ == '__main__':
    local_comm = LocalCommunication.create_from_config()
    ssh_comm = SshCommunication.create_from_config('atmlxint2')
    res = Research.open('RC_MOEHLIS', ssh_comm)

    n_ics = 3
    data = {
        'res': res,
        'time_step': 0.001,
        'final_time': 20000,
        'save_dt': 1,
        're': 275,
        'l_x': 1.75,
        'l_z': 1.2,
        'equation_type': 'full',
        'stop_condition': 'close_to_laminar_flow',
        'initial_conditions': [list(generate_random_ic_for_lifetime_distr()) for _ in range(n_ics)],
        'input_filename': 'inputs.json',
        'output_filenames': [str(i) for i in range(1, n_ics + 1)],
        'description': 'Long Moehlis (possible not) relaminarising trajectories for testing perturbation equations'
    }

    graph = LocalPythonTimeIntegrationGraph(res, local_comm,
                                            MoehlisModelIntegrator(input_filename_key='input_filename', nohup=True),
                                            input_filename=data['input_filename'],
                                            task_prefix=f'ESNPrediction')

#    graph = RemotePythonTimeIntegrationGraph(res, ssh_comm,
#                                             MoehlisModelIntegrator(input_filename_key='input_filename', nohup=True),
#                                             input_filename=data['input_filename'],
#                                             task_prefix='PertMoehlisTimeIntegration')
    okay = graph.run(data)
    if not okay:
        print(data['__EXCEPTION__'])
    print(data)
