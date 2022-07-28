import math
import time
import posixpath

import numpy as np

from restools.standardised_programs import StandardisedIntegrator, StandardisedProgramEdge
from comsdk.research import CreateTaskGraph
from comsdk.graph import Graph, State, Func
from comsdk.edge import Edge, dummy_predicate, InOutMapping, DownloadFromRemoteEdge, \
    UploadOnRemoteEdge, make_dump, make_cd


def relaminarisation_time(ke, T=1000, debug=False):
    '''
    We detect turbulent-to-laminar transition if the kinetic energy is larger than 15 for more than T time units
    and return relaminarisation time is this event has occured. Otherwise None is returned
    '''
    transition_start = None
    for t in range(len(ke)):
        if transition_start:
            if t - transition_start > T:
                if debug:
                    print(f'Found turbulent-to-laminar transition from {transition_start} to {t}')
                return t
            if ke[t] < 15:
                transition_start = None
        elif ke[t] > 15:
            transition_start = t
    last_t = len(ke) - 1
    if debug and transition_start is not None:
        print(f'Found turbulent-to-laminar transition from {transition_start} to infty ({last_t})')
    return last_t if transition_start is not None else None


def is_laminarised(ke, T=1000, debug=False):
    '''
    We detect turbulent-to-laminar transition if the kinetic energy is larger than 15 for more than T time units
    '''
    transition_start = None
    for t in range(len(ke)):
        if transition_start:
            if t - transition_start > T:
                if debug:
                    print(f'Found turbulent-to-laminar transition from {transition_start} to {t}')
                return True
            if ke[t] < 15:
                transition_start = None
        elif ke[t] > 15:
            transition_start = t
    return False


def survival_function(data, debug=False):
    values = sorted([t for t in data if not math.isnan(t)])
    if debug and len(data) != len(values):
        print(f'While building survival function, filtered {len(data) - len(values)} "None" points')
    probs = np.array([1 - i/len(values) for i in range(len(values))])
    return values, probs


def laminarization_probability(model, trajs):
    n = len(trajs)
    n_lam = 0
    for ens_member in trajs:
        ke = model.kinetic_energy(ens_member[:-1, :])
        if np.all(ke > 10):
            n_lam += 1
    p_lam = float(n_lam) / n
    return p_lam


def generate_random_ic_for_lifetime_distr():
    # this algorithm is from Moehlis 2004
    ic = np.random.uniform(-1, 1, size=9)
    k = np.sum(ic**2)
    return np.sqrt(0.3 / k) * ic


def generate_random_ic_for_laminarization_probability(model, energy_level):
    ic = np.random.uniform(-0.5, 0.5, size=9)
    ke_raw = model.kinetic_energy(ic)
    coeff = np.sqrt(energy_level / ke_raw)
    return coeff * ic


class LocalPythonTimeIntegrationGraph(Graph):
    def __init__(self,
                 res,
                 comm,
                 integrator,
                 input_filename,
                 task_prefix=''):
        def task_name_maker(d):
            task_name = task_prefix
            task_name += '_R_{}'.format(d['re'])
            task_name += '_T_{}'.format(d['final_time'])
            task_name += '_ens_{}'.format(len(d['initial_conditions']))
            return task_name
        task_start, task_end = CreateTaskGraph.create_branch(res, task_name_maker=task_name_maker)
        ti_init, ti_term = LocalPythonTimeIntegrationGraph.create_branch(comm, integrator)
        dumping_edge = Edge(dummy_predicate, Func())
        task_end.connect_to(ti_init, edge=dumping_edge)
        dumping_edge.postprocess = make_dump(input_filename, omit=['res'], method='json')
        super().__init__(task_start, ti_term)

    @staticmethod
    def create_branch(comm,
                      integrator,
                      relative_keys=(),
                      keys_mapping={},
                      array_keys_mapping=None,
                      cd_path_key=None):
        s_init = State('READY_FOR_PYTHON_TIMEINTEGRATION', array_keys_mapping=array_keys_mapping)
        s_term = State('PYTHON_TIMEINTEGRATION_FINISHED')
        integrate_edge = StandardisedProgramEdge(integrator,
                                                 comm, relative_keys=relative_keys, keys_mapping=keys_mapping)
        if cd_path_key:
            integrate_edge.use_proxy_data_for_pre_post_processing = True
            integrate_edge.preprocess=make_cd(cd_path_key)
            integrate_edge.postprocess=make_cd('..')
        s_init.connect_to(s_term, edge=integrate_edge)
        return s_init, s_term


class RemotePythonTimeIntegrationGraph(Graph):
    def __init__(self,
                 res,
                 remote_comm,
                 integrator,
                 input_filename,
                 task_prefix='',
                 output_filenames_key='output_filenames'):
        def task_name_maker(d):
            task_name = task_prefix
            task_name += '_R_{}'.format(d['re'])
            if 'final_time' in d:
                task_name += '_T_{}'.format(d['final_time'])
            if 'initial_conditions' in d:
                task_name += '_ens_{}'.format(len(d['initial_conditions']))
            return task_name

        task_start, task_end = CreateTaskGraph.create_branch(res, task_name_maker=task_name_maker, remote=True)
        ti_start, ti_end = RemotePythonTimeIntegrationGraph.create_branch(remote_comm, integrator,
                                                                          output_filenames_key=output_filenames_key)
        dumping_edge = Edge(dummy_predicate, Func())
        dumping_edge.postprocess = make_dump(input_filename, omit=['res'], method='json')
        task_end.connect_to(ti_start, edge=dumping_edge)
        super().__init__(task_start, ti_end)

    @staticmethod
    def create_branch(remote_comm,
                      integrator_prog: StandardisedIntegrator,
                      relative_keys=(),
                      keys_mapping={},
                      array_keys_mapping=None,
                      init_field_at_remote_key=None,
                      output_filenames_key='output_filenames',
                      cd_path_key=None,
                      extra_keys_to_upload=()):
        def task_finished(d):
            time.sleep(2)
            #return '__finished__' in remote_comm.listdir(d['__REMOTE_WORKING_DIR__'])
            return '__finished__' in remote_comm.listdir(posixpath.join(d['__REMOTE_WORKING_DIR__'],
                                                                        d['optimal_esn_filename']))

        task_finished_predicate = Func(func=task_finished)
        task_not_finished_predicate = Func(func=lambda d: not task_finished(d))
        io_mapping = InOutMapping(relative_keys=relative_keys, keys_mapping=keys_mapping)
        upload_edge = UploadOnRemoteEdge(remote_comm,
                                         local_paths_keys=tuple(set(extra_keys_to_upload + integrator_prog.trailing_args_keys)),
                                         already_remote_path_key=init_field_at_remote_key)
        integrate_edge = StandardisedProgramEdge(integrator_prog, remote_comm,
                                                 io_mapping=io_mapping, remote=True)
        download_edge = DownloadFromRemoteEdge(remote_comm,
                                               predicate=task_finished_predicate,
                                               io_mapping=io_mapping,
                                               remote_paths_keys=(output_filenames_key,),
                                               update_paths=False)
        if cd_path_key:
            integrate_edge.use_proxy_data_for_pre_post_processing = True
            integrate_edge.preprocess=make_cd('optimal_esn_filename')
            integrate_edge.postprocess=make_cd('..')
        s_ready = State('READY_FOR_TIME_INTEGRATION', array_keys_mapping=array_keys_mapping)
        s_uploaded_input_files = State('UPLOADED_INPUT_FILES')
        s_integrated = State('INTEGRATED')
        s_downloaded_output_files = State('DOWNLOADED_OUTPUT_FILES')
        s_ready.connect_to(s_uploaded_input_files, edge=upload_edge)
        s_uploaded_input_files.connect_to(s_integrated, edge=integrate_edge)
        s_integrated.connect_to(s_downloaded_output_files, edge=download_edge)
        s_integrated.connect_to(s_integrated, edge=Edge(task_not_finished_predicate, Func(func=lambda d: time.sleep(5))))
        return s_ready, s_downloaded_output_files
