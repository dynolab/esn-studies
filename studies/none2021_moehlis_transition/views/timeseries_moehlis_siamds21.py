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

    task = 1
    res_id = 'RC_MOEHLIS'
    res = Research.open(res_id)
    task_path = res.get_task_path(task)
    ens_members = [1, 8, 5, 10]  # 3, 5
    #ens_members = [8, 9, 10, 10]  # 3, 5
    with open(os.path.join(task_path, 'inputs.json'), 'r') as f:
        inputs = json.load(f)
        m = MoehlisFaisstEckhardtModel(Re=inputs['re'], L_x=inputs['l_x'] * np.pi, L_z=inputs['l_z'] * np.pi)
    lam_state_ke = m.kinetic_energy(m.laminar_state)
    fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True)
    for ax, m_i in zip(axes, ens_members):
        with open(os.path.join(task_path, str(m_i)), 'rb') as f:
            data = pickle.load(f)
            t = data['time']
            ts = data['timeseries']
        ke = m.kinetic_energy(ts)
        ax.plot(t, ke, linewidth=3)
        ax.set_ylabel(r'$E$')
        ax.grid()
    t_lims = axes[0].get_xlim()
    for ax in axes:
        ax.plot([0, t_lims[1]], [lam_state_ke, lam_state_ke], 'k--', linewidth=3, zorder=-100)
    axes[-1].set_xlabel(r'$t$')
    plt.tight_layout()
    plt.savefig('timeseries_moehlis.eps', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    plt.show()
