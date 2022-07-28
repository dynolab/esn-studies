import os
import sys
sys.path.append(os.getcwd())
import time
import json
import posixpath

import numpy as np

from restools.standardised_programs import EsnIntegrator, EsnTrainer
from restools.timeintegration import TimeIntegrationLowDimensional
from studies.none2021_predicting_transition_using_reservoir_computing.extensions import LocalPythonTimeIntegrationGraph, RemotePythonTimeIntegrationGraph
from comsdk.communication import LocalCommunication, SshCommunication
from comsdk.research import Research, CreateTaskGraph
from comsdk.graph import Graph, State, Func
from comsdk.edge import Edge, dummy_predicate, make_dump, make_composite_func, make_cd, make_mkdir
import comsdk.misc as aux


def make_check_job_finished(comm=None):
    def _check_job_finished(d):
        job_finished = False
        remote = '__REMOTE_WORKING_DIR__' in d
        if remote:
            job_finished = '__finished__' in comm.listdir(d['__REMOTE_WORKING_DIR__'])
        else:
            job_finished = os.path.exists(
                os.path.join(d['__WORKING_DIR__'],
                             d['finished'])
            )
        if not job_finished:
            time.sleep(2)
        return job_finished
    return _check_job_finished

def make_add_esn_path_to_data(remote=False):
    def _add_esn_path_to_data(d):
        wd = '__REMOTE_WORKING_DIR__' if remote else '__WORKING_DIR__'
        data['esn_path'] = os.path.join(d[wd],
                                        d['optimal_esn_filename'],
                                        d['optimal_esn_filename'])
    return _add_esn_path_to_data


def make_optimal_esn_filename(d1, d2):
    d1['optimal_esn_filename'] = d2['optimal_esn_filename']


def make_little_dump(input_filename, omit=None, chande_dir=True, remote=False):
    def _little_dump(d):
        if omit is None:
            dumped_d = d
        else:
            if (isinstance(d, aux.ProxyDict)):
                dumped_d = {key: val for key, val in d._data.items() if not key in omit}
            else:
                dumped_d = {key: val for key, val in d.items() if not key in omit}
        dumped_d['training_input_filename'] = d['training_input_filename']
        dumped_d['ti_input_filename'] = d['ti_input_filename']
        dumped_d['random_seed_starts_at'] = d['random_seed_starts_at']
        dumped_d['optimal_esn_filename'] = d['optimal_esn_filename']
        dumped_d['training_output_filenames'] = d['training_output_filenames']
        dumped_d['output_filenames'] = d['output_filenames']
        if remote:
            dumped_d['training_timeseries_path'] = posixpath.join(
                d['__REMOTE_WORKING_DIR__'],
                os.path.basename(d['training_timeseries_path'])
            )
            dumped_d['test_timeseries_path'] = posixpath.join(
                d['__REMOTE_WORKING_DIR__'],
                os.path.basename(d['test_timeseries_path'])
            )
        else:
            dumped_d['training_timeseries_path'] = d['training_timeseries_path']
            dumped_d['test_timeseries_path'] = d['test_timeseries_path']
        dumped_d['finished'] = d['finished']
        dumped_d['started'] = d['started']
        if (chande_dir):
            make_cd('optimal_esn_filename')(dumped_d)
        dump_path = os.path.join(dumped_d['__WORKING_DIR__'],
                                 d[input_filename])
        with open(dump_path, 'w') as f:
            json.dump(dumped_d, f)
        if (chande_dir):
            make_cd('..')(dumped_d)
    return _little_dump


class ESNTrainAndIntergateGraph(Graph):
    def __init__(self, res, comm, input_filename, task_prefix='', remote=False):
        def task_name_maker(d):
            task_name = task_prefix
            task_name += '_R_{}'.format(d['re'])
            task_name += '_T_{}'.format(d['final_time'])
            task_name += '_ens_{}'.format(len(d['initial_conditions']))
            task_name += '_nESNs_{}'.format(d['n_ESNs'])
            return task_name
        # (task_start) -> (task_end) -пустое ребро-> (tt_init) -train-> (tt_term) -fin-> (ti_init) -integrate-> (ti_term)
        #                                                               (not fin)^
        #                                          ^тут начинается subgraph
        #        state_for_keys_mapping = State('START Making implicit parallelization', array_keys_mapping={'input_filename':'input_filenames', 'optimal_esn_filename':'optimal_esn_filenames'})
        #!   #, array_keys_mapping={'input_filename':'input_filenames', 'optimal_esn_filename':'optimal_esn_filenames', 'output_filenames':'output_filenames', 'finished':'finished', 'started':'started', 'pipes_index':'pipes_index'})
        state_for_keys_mapping = State('START Making implicit parallelization',
                                       array_keys_mapping={
                                           'training_input_filename': 'training_input_filenames',
                                           'ti_input_filename': 'ti_input_filenames',
                                           'optimal_esn_filename': 'optimal_esn_filenames',
                                           'training_timeseries_path': 'training_timeseries_path',
                                           'test_timeseries_path': 'test_timeseries_path',
                                           'pipes_index': 'pipes_index',
                                           'random_seed_starts_at': 'random_seed_starts_at'})
        state_for_cd_esn_dir = State('START Create dir for esn')
        state_for_optimal_esn_filename = State('START Making names for esn files')
        
        if remote:
            tt_init, tt_term = RemotePythonTimeIntegrationGraph.create_branch(
                comm,
                EsnTrainer(input_filename_key='training_input_filename',
                           nohup=True,
                           pipes_index_key='pipes_index'),
                output_filenames_key='training_output_filenames',
                cd_path_key='optimal_esn_filename',
                extra_keys_to_upload=('training_timeseries_path', 'test_timeseries_path'),
            )#, array_keys_mapping={'input_filename':'input_filenames', 'optimal_esn_filename':'optimal_esn_filenames'})
            ti_init, ti_term = RemotePythonTimeIntegrationGraph.create_branch(
                comm,
                EsnIntegrator(input_filename_key='ti_input_filename',
                              nohup=True,
                              pipes_index_key='pipes_index'),
                cd_path_key='optimal_esn_filename',
            )
        else:
            tt_init, tt_term = LocalPythonTimeIntegrationGraph.create_branch(
                comm,
                EsnTrainer(input_filename_key='training_input_filename',
                           nohup=True,
                           pipes_index_key='pipes_index'),
                cd_path_key='optimal_esn_filename',
            )#, array_keys_mapping={'input_filename':'input_filenames', 'optimal_esn_filename':'optimal_esn_filenames'})
            ti_init, ti_term = LocalPythonTimeIntegrationGraph.create_branch(
                comm,
                EsnIntegrator(input_filename_key='ti_input_filename',
                              nohup=True,
                              pipes_index_key='pipes_index'),
                cd_path_key='optimal_esn_filename',
            )
    

        dummy_edge = Edge(dummy_predicate, Func())
        edge_esn_dir = Edge(dummy_predicate, Func())
        edge_esn_name = Edge(dummy_predicate, Func())
        
        job_finished_predicate = Func(
            func=make_composite_func(
                make_cd('optimal_esn_filename'),
                make_check_job_finished(comm=comm),
                make_cd('..')
            )
        )
        job_unfinished_predicate = Func(func=lambda d: not job_finished_predicate.func(d))
        #lambda d: check_job_finished(d))
        notdone_edge = Edge(job_unfinished_predicate, Func())
        done_edge = Edge(job_finished_predicate, Func())
        
        state_for_keys_mapping.connect_to(state_for_optimal_esn_filename, edge=dummy_edge)
        state_for_optimal_esn_filename.connect_to(state_for_cd_esn_dir, edge=edge_esn_name)
        state_for_cd_esn_dir.connect_to(tt_init, edge=edge_esn_dir)
        
        tt_term.connect_to(tt_term, edge=notdone_edge)
        tt_term.connect_to(ti_init, edge=done_edge)
        
        edge_esn_dir.use_proxy_data_for_pre_post_processing = True
        edge_esn_name.use_proxy_data_for_pre_post_processing = True
        done_edge.use_proxy_data_for_pre_post_processing = True

        edge_esn_dir.preprocess = make_composite_func(
            make_mkdir(key_path='optimal_esn_filename',
                       remote_comm=comm),
            make_little_dump('training_input_filename',
                             omit=['res'],
                             remote=remote),
            make_add_esn_path_to_data(remote),
            make_little_dump('ti_input_filename',
                             omit=['res'],
                             remote=remote),
        )
        edge_esn_name.postprocess = make_composite_func(
            make_little_dump('training_input_filename',
                             omit=['res'],
                             chande_dir=False,
                             remote=remote),
            make_add_esn_path_to_data(remote),
            make_little_dump('ti_input_filename',
                             omit=['res'],
                             chande_dir=False,
                             remote=remote),
        )
     
        #train_edge.use_proxy_data_for_pre_post_processing = True
        #integrate_edge.use_proxy_data_for_pre_post_processing = True
        #train_edge.preprocess=make_cd('optimal_esn_filename')
        #train_edge.postprocess=make_cd('..')
        #integrate_edge.preprocess=make_cd('optimal_esn_filename')
        #integrate_edge.postprocess=make_cd('..')        
        done_edge.postprocess = make_composite_func(
            make_add_esn_path_to_data(remote),
            #make_little_dump('training_input_filename',
            #                 omit=['res'],
            #                 remote=remote),
            make_little_dump('ti_input_filename',
                             omit=['res'],
                             remote=remote),
        )
        
        subgraph = Graph(state_for_keys_mapping, ti_term)
        
        # (task_start) -> (task_end) -пустое ребро->  (subgraph_state)   
        task_start, task_end = CreateTaskGraph.create_branch(res,
                                                             task_name_maker=task_name_maker,
                                                             remote=remote)#, array_keys_mapping={'input_filename':'input_filename'})
        subgraph_state = State('START Working with ESN')
        dumping_edge = Edge(dummy_predicate, Func())
        task_end.connect_to(subgraph_state, edge=dumping_edge)
        dumping_edge.postprocess = make_composite_func(
            make_dump(input_filename,
                      omit=['res'],
                      method='json')
        )

        subgraph_state.replace_with_graph(subgraph)
        super().__init__(task_start, ti_term)


if __name__ == '__main__':
#    local_comm = LocalCommunication.create_from_config()
    ssh_comm = SshCommunication.create_from_config('hpc_rk6')
    res = Research.open('RC_MOEHLIS', ssh_comm)
#    res = Research.open('RC_MOEHLIS')

   # ics = [[0,1,2,3,4,5,6,7]]        #initial condition for all ESNs
    ics = []
    source_tasks = [42, 45]
    for source_task in source_tasks:
        for ic_i in range(1, 100+1):
            ti = TimeIntegrationLowDimensional(res.get_task_path(source_task), ic_i)
            #q = ti.timeseries - np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float64)
            #ti = TimeIntegrationLowDimensional(res.get_task_path(source_task), 1)
            ics.append(ti.timeseries[:10].tolist())
            #ics.append(ti.timeseries[begin_time-10:begin_time].tolist())
    n_ics = len(ics)
    #begin_time = 13940 + 100 + 100 + 100
    n_ESNs = 10     #number of ESN 
   
    re = 275
    esn_name = f'esn_re_{re}'
    #esn_name = 'esn_trained_wo_lam_event'
    l_x = 1.75
    l_z = 1.2
    #n_steps = 20000
    n_steps = 20000

    task = res._get_next_task_number()
    res._tasks_number -= 1
    task_prefix=f'ESNEnsemble'
    optimal_esn_filenames = [f'{esn_name}_{i}' for i in range(1, n_ESNs + 1)]
    
    data = {
        'res': res,
        'pipes_index': [str(i) for i in range(1, n_ESNs + 1)],
        'n_ESNs': n_ESNs,
        
        'training_timeseries_path': n_ESNs*[os.path.join(res.local_research_path, f'training_timeseries_re_{re}')],
        'test_timeseries_path': n_ESNs*[os.path.join(res.local_research_path, f'test_timeseries_re_{re}')],
        'synchronization_len': 10,
        'test_chunk_timeseries_len': 300,
        'spectral_radius_values': [0.5],
        #'sparsity_values': [0.1, 0.5, 0.9],
        'sparsity_values': [0.9],
        'trial_number': 1,
        'random_seed_starts_at': list(range(60 + 1, n_ESNs + 60 + 1)),
        'reservoir_dimension': 1500,
        'optimal_esn_filenames': optimal_esn_filenames,
        'optimal_esn_filename': f'{esn_name}',
        'finished': f'__finished__',
        'started': f'__started__',
#        'finished': [f'__finished_{i}__' for i in range(1, n_ESNs + 1)],
#        'started': [f'__started_{i}__' for i in range(1, n_ESNs + 1)],


        'dt_as_defined_by_esn': 1,
        'n_steps': n_steps,
        'final_time': n_steps,
        're': re,
        'l_x': l_x,
        'l_z': l_z,
        'initial_conditions': ics,
        'input_filename': 'inputs.json',
        'training_input_filenames': [ f'training_inputs_{i}.json' for i in range(1, n_ESNs + 1)],
        'ti_input_filenames': [ f'ti_inputs_{i}.json' for i in range(1, n_ESNs + 1)],
        #'training_output_filenames': [f'{n}/{n}' for n in optimal_esn_filenames],
        #'output_filenames': str(1),
        'output_filenames': [str(i) for i in range(1, n_ics + 1)],
        'training_output_filenames': None,
        #'output_filenames': None,
        #'output_filenames': [str(i) for i in range(1, n_ESNs + 1)],
        'description': f'Predictions of trained ESN ensemble with one initial condition for all ESNs. '
                       f'Noise is disabled while predicting'
    }

#    graph = ESNTrainAndIntergateGraph(res,
#                                      local_comm,
#                                      input_filename=data['input_filename'],
#                                      task_prefix=task_prefix,
#                                      remote=False)

    graph = ESNTrainAndIntergateGraph(res,
                                      ssh_comm,
                                      input_filename=data['input_filename'],
                                      task_prefix=task_prefix,
                                      remote=True)

 #   print("graph")
    okay = graph.run(data)
    if not okay:
        print(data['__EXCEPTION__'])
 #   print(data)