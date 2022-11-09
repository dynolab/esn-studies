import os
import sys
sys.path.append(os.getcwd())

import numpy as np

from restools.timeintegration import TimeIntegrationLowDimensional
from studies.none2021_moehlis_transition.extensions import ESNTrainAndIntergateGraph
from comsdk.communication import LocalCommunication, SshCommunication
from comsdk.research import Research


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
        'random_seed_starts_at': list(range(90 + 1, n_ESNs + 90 + 1)),
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