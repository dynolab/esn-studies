import os
import sys
sys.path.append(os.getcwd())
import time
import json

import numpy as np

from restools.standardised_programs import StandardisedIntegrator, StandardisedProgramEdge, MoehlisModelIntegrator
from studies.none2021_moehlis_transition.extensions import LocalPythonTimeIntegrationGraph,\
    RemotePythonTimeIntegrationGraph, generate_random_ic_for_laminarization_probability
from comsdk.communication import LocalCommunication, SshCommunication
from comsdk.research import Research, CreateTaskGraph
from thequickmath.reduced_models.transition_to_turbulence import MoehlisFaisstEckhardtModel


if __name__ == '__main__':
    local_comm = LocalCommunication.create_from_config()
#    ssh_comm = SshCommunication.create_from_config('atmlxint2')
#    res = Research.open('RC_MOEHLIS', ssh_comm)
    res = Research.open('RC_MOEHLIS')

    base_task = 83
    with open(os.path.join(res.get_task_path(base_task), 'inputs.json'), 'r') as f:
        inputs = json.load(f)
    m = MoehlisFaisstEckhardtModel(Re=inputs['re'], L_x=inputs['l_x'] * np.pi, L_z=inputs['l_z'] * np.pi)
    print(inputs.keys())
    n_rp_per_energy_level = 30
    rps = []
    ics = []
    trajectory_numbers = []
    for energy_i in range(len(inputs['energy_levels'])):
        rps.append([])
        trajectory_numbers.append([])
        for rp_i in range(n_rp_per_energy_level):
            rp = generate_random_ic_for_laminarization_probability(m, inputs['energy_levels'][energy_i])
            rps[-1].append(rp.tolist())
            ic = m.laminar_state + rp
            ics.append(ic.tolist())
            trajectory_numbers[-1].append(len(ics))

    n_ics = len(ics)
    data = {
        'res': res,
        'time_step': 0.001,
        'final_time': 300,
        'save_dt': 1,
        're': inputs['re'],
        'l_x': inputs['l_x'],
        'l_z': inputs['l_z'],
        'equation_type': 'full',
        'stop_condition': 'default',
        'trajectory_numbers': trajectory_numbers,
        'energy_levels': inputs['energy_levels'],
        'rps': rps,
        'initial_conditions': ics,
        'input_filename': 'inputs.json',
        'output_filenames': [str(i) for i in range(1, n_ics + 1)],
        'description': 'Short Moehlis trajectories for computing laminarization probability'
    }

    graph = LocalPythonTimeIntegrationGraph(res, local_comm,
                                            MoehlisModelIntegrator(input_filename_key='input_filename', nohup=True),
                                            input_filename=data['input_filename'],
                                            task_prefix=f'MoehlisLaminarizationProbabilityStudy')

#    graph = RemotePythonTimeIntegrationGraph(res, ssh_comm,
#                                             MoehlisModelIntegrator(input_filename_key='input_filename', nohup=True),
#                                             input_filename=data['input_filename'],
#                                             task_prefix='PertMoehlisTimeIntegration')
    okay = graph.run(data)
    if not okay:
        print(data['__EXCEPTION__'])
    print(data)
