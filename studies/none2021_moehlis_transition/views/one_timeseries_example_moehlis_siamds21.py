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

    # 33, 4 has two awful features: some exmtreme oscillations and delaminarization!!!

    task = 33
    res_id = 'RC_MOEHLIS'
    res = Research.open(res_id)
    task_path = res.get_task_path(task)
    ens_member = 1
    with open(os.path.join(task_path, 'inputs.json'), 'r') as f:
        inputs = json.load(f)
        m = MoehlisFaisstEckhardtModel(Re=inputs['re'], L_x=inputs['l_x'] * np.pi, L_z=inputs['l_z'] * np.pi)
    with open(os.path.join(task_path, str(ens_member)), 'rb') as f:
        data = pickle.load(f)
        t = data['time'][1000:7000] - 1000.
        ts = data['timeseries'][1000:7000]
    fig, axes = plt.subplots(5, 2, figsize=(7, 8))
    for i in range(9):
        ax = axes[int(i//2), i - 2*int(i//2)]
        ax.plot(t, ts[:, i], linewidth=1)
        ax.set_ylabel(r'$a_{' + str(i+1) + r'}$')
        ax.grid()
    axes[-1, -1].axis('off')
#    t_lims = axes[0].get_xlim()
#    for ax in axes:
#        ax.plot([0, t_lims[1]], [lam_state_ke, lam_state_ke], 'k--', linewidth=3)
#    axes[-1].set_xlabel(r'$t$')
    plt.tight_layout()
    plt.savefig('one_timeseries_example_moehlis.eps', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    plt.show()
