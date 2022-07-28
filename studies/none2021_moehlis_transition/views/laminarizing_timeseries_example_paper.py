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


def plot_trajectories(axes, res, trajs, color=None, linewidth=3):
    for ax, traj_data in zip(axes, trajs):
        task_path = res.get_task_path(traj_data['task'])
        with open(os.path.join(task_path, str(traj_data['i'])), 'rb') as f:
            data = pickle.load(f)
            t = data['time'][:traj_data['end_time']]
            ts = data['timeseries'][:traj_data['end_time']]
        ke = m.kinetic_energy(ts)
        lines = ax.plot(t, ke, linewidth=linewidth, color=color)
        if 'continue_until' in traj_data:
            ax.plot([t[-1] + t[-1] - t[-2], traj_data['continue_until']], [ke[-1], ke[-1]], linewidth=linewidth, color=color)
        print(lines[0].get_color())


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    res_id = 'RC_MOEHLIS'
    res = Research.open(res_id)
    trajs_esn = [
        {'task': 19, 'i': 10, 'end_time': 8000},
#        {'task': 18, 'i': 8, 'end_time': 5000},
#        {'task': 19, 'i': 5, 'end_time': 4000},
#        {'task': 17, 'i': 4, 'end_time': 3000},
    ]
    trajs_moehlis = [
        {'task': 1, 'i': 1, 'end_time': -1, 'continue_until': 8000},
#        {'task': 1, 'i': 8, 'end_time': -1},
#        {'task': 1, 'i': 5, 'end_time': -1},
#        {'task': 1, 'i': 10, 'end_time': -1},
    ]
    task_path = res.get_task_path(trajs_esn[0]['task'])
    #ens_members = [9, 10, 7, 8]  # 17, 4 = 1500,      18, 8 = 4000,     19, 5 = 2800      19, 10 = 5400
    with open(os.path.join(task_path, 'inputs.json'), 'r') as f:
        inputs = json.load(f)
        m = MoehlisFaisstEckhardtModel(Re=inputs['re'], L_x=inputs['l_x'] * np.pi, L_z=inputs['l_z'] * np.pi)
    fig, axes = plt.subplots(1, 1, figsize=(8.5, 3), sharex=True)
    axes = [axes]
    plot_trajectories(axes, res, trajs_moehlis, color='#ccdeea', linewidth=3)
    plot_trajectories(axes, res, trajs_esn)
    t_lims = axes[0].get_xlim()
    lam_state_ke = m.kinetic_energy(m.laminar_state)
    for ax in axes:
        #ax.plot([0, t_lims[1]], [lam_state_ke, lam_state_ke], 'k--', linewidth=3)
        ax.set_ylabel(r'$E$')
        ax.grid()
    axes[-1].set_xlabel(r'$t$')
    plt.tight_layout()
    plt.savefig('laminarizing_timeseries_example_paper.svg', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    plt.show()
