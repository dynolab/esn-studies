import os
import sys
sys.path.append(os.getcwd())
import pickle
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import restools
from studies.none2021_moehlis_transition.extensions import relaminarisation_time, survival_function
from comsdk.research import Research
from comsdk.misc import load_from_json, find_all_files_by_named_regexp
from thequickmath.reduced_models.transition_to_turbulence import MoehlisFaisstEckhardtModel


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    res_id = 'RC_MOEHLIS'
    res = Research.open(res_id)
    data = {
        'Truth': {
            're_values': [200, 250, 300],
            'tasks': [
                [12, 13],
                [11, 14],
                list(range(1, 11)) + [15]
            ],

        },
        'ESN': {
            're_values': [250, 300],
            'tasks': [
                [29, 30],
                list(range(17, 28)),
            ],
        }
    }
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for model_type, d in data.items():
        for re, tasks in zip(d['re_values'], d['tasks']):
            relam_times = []
            for t_i in range(len(tasks)):
                task_path = res.get_task_path(tasks[t_i])
                with open(os.path.join(task_path, 'inputs.json'), 'r') as f:
                    inputs = json.load(f)
                    m = MoehlisFaisstEckhardtModel(Re=re, L_x=inputs['l_x'] * np.pi, L_z=inputs['l_z'] * np.pi)
                filename_and_params = find_all_files_by_named_regexp(r'^(?P<num>\d+)$', task_path)
                file_paths = [os.path.join(task_path, filename) for filename, params in filename_and_params]
                for file_path in file_paths:
                    with open(file_path, 'rb') as f:
                        data_ = pickle.load(f)
                        t = data_['time']
                        ts = data_['timeseries']
                    ke = m.kinetic_energy(ts)
                    relam_times.append(relaminarisation_time(ke, T=1000, debug=False))
            ax.semilogy(*survival_function(relam_times), 'o--', linewidth=2, label=r'$Re = ' + str(re) + r'$' + f' ({model_type})')
    ax.grid()
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$S(t)$')
    ax.legend()
    plt.tight_layout()
    #plt.savefig('phase_space.png', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    plt.show()
