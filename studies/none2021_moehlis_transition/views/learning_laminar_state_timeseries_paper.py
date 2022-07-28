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

    data = [
        {'task': 73, 'end_time': 3100},
        {'task': 63, 'end_time': 3200},
        {'task': 66, 'end_time': 3300},
        {'task': 60, 'end_time': 3700},
    ]

    with open(os.path.join(res.local_research_path, 'training_timeseries_re_275'), 'rb') as f:
        d = pickle.load(f)
        original_t = d['time']
        original_ts = d['timeseries']
    with open(os.path.join(res.get_task_path(data[0]['task']), 'inputs.json'), 'r') as f:
        inputs = json.load(f)
        m = MoehlisFaisstEckhardtModel(Re=inputs['re'], L_x=inputs['l_x'] * np.pi, L_z=inputs['l_z'] * np.pi)
    original_ke = m.kinetic_energy(original_ts)

    fig, axes = plt.subplots(len(data), 2, figsize=(6, 5))
    for i, d in enumerate(data):
        task_path = res.get_task_path(d['task'])
        axes[i][0].plot(original_t, original_ke, linewidth=2)
        axes[i][0].fill_between([0, d['end_time']], y1=-1, y2=23, color='#ccc')
        #for j in range(1, 11):
        for j in range(30, 41):
            with open(os.path.join(task_path, str(j)), 'rb') as f:
                d_ = pickle.load(f)
                t = d_['time']
                ts = d_['timeseries']
            ke = m.kinetic_energy(ts)
            axes[i][1].plot(t, ke, linewidth=2)
        for j in range(2):
            axes[i][j].set_ylim((0, 22))
            if i != len(data) - 1:
                axes[i][j].set_xticklabels([])
            if j == 1:
                axes[i][j].set_yticklabels([])
            axes[i][j].grid()
    axes[0][0].set_title('Training timeseries')
    axes[0][1].set_title('ESN predictions')
    plt.tight_layout(h_pad=0.2)
    plt.savefig('learning_laminar_state_timeseries.eps', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    plt.show()
