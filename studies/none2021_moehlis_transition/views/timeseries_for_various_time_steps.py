import os
import sys
sys.path.append(os.getcwd())
import pickle
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import restools
from comsdk.research import Research
from comsdk.misc import load_from_json
from thequickmath.reduced_models.transition_to_turbulence import MoehlisFaisstEckhardtModel


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    res_id = 'RC_MOEHLIS'
    res = Research.open(res_id)
    tasks = [137, 138, 139, 140]
    timesteps = [2, 3, 4, 5]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for task_i in range(len(tasks)):        
        task_path = res.get_task_path(tasks[task_i])
        with open(os.path.join(task_path, '1'), 'rb') as f:
            data = pickle.load(f)
            t = data['time']
            ts = data['timeseries']
        with open(os.path.join(task_path, 'inputs.json'), 'r') as f:
            inputs = json.load(f)
            m = MoehlisFaisstEckhardtModel(Re=inputs['re'], L_x=inputs['l_x'] * np.pi, L_z=inputs['l_z'] * np.pi)
        ke = m.kinetic_energy(ts)
        ax.semilogy(t, ke, 'o--', linewidth=2, label=r'$\triangle t_{ESN} = ' + str(timesteps[task_i]) + r'$')
        #ax.set_xlim((-100, 7000))
        #ax.set_ylim((0, 22))
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$E$')
    ax.legend(fontsize=16, loc='lower right')
    ax.grid()
    plt.tight_layout()
    plt.savefig('predictions_with_different_timesteps.eps', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    plt.show()
