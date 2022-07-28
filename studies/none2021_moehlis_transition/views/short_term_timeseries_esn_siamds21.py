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


def plot_trajectories(axes, res, trajs, time_shift=0, cut_off_time=-1, color=None, linewidth=3):
    for ax, traj_data in zip(axes, trajs):
        task_path = res.get_task_path(traj_data['task'])
        with open(os.path.join(task_path, str(traj_data['i'])), 'rb') as f:
            data = pickle.load(f)
            t = data['time'][:cut_off_time] + time_shift
            ts = data['timeseries'][:cut_off_time]
        ke = m.kinetic_energy(ts)
        ax.plot(t, ke, linewidth=linewidth, color=color)
        if time_shift != 0:
            ax.plot(t[0], ke[0], 'ro', markersize=10)


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    res_id = 'RC_MOEHLIS'
    res = Research.open(res_id)
    trajs_esn = [
        {'task': 17, 'i': 1},
        {'task': 17, 'i': 8},
        {'task': 17, 'i': 5},
        {'task': 17, 'i': 10},
    ]
    trajs_moehlis = [
        {'task': 1, 'i': 1},
        {'task': 1, 'i': 8},
        {'task': 1, 'i': 5},
        {'task': 1, 'i': 10},
    ]
    task_path = res.get_task_path(trajs_esn[0]['task'])
    with open(os.path.join(task_path, 'inputs.json'), 'r') as f:
        inputs = json.load(f)
        m = MoehlisFaisstEckhardtModel(Re=inputs['re'], L_x=inputs['l_x'] * np.pi, L_z=inputs['l_z'] * np.pi)
    fig, axes = plt.subplots(4, 1, figsize=(12, 7.5), sharex=True, sharey=True)
    plot_trajectories(axes, res, trajs_moehlis, cut_off_time=500, color='#ccdeea', linewidth=3)
    plot_trajectories(axes, res, trajs_esn, time_shift=10, cut_off_time=500)
    t_lims = axes[0].get_xlim()
    for ax in axes:
        #ax.set_ylim((0, 11.5))
        ax.set_ylabel(r'$E$')
        ax.grid()
    axes[-1].set_xlabel(r'$t$')
    plt.tight_layout()
    plt.savefig('short_term_timeseries_esn.eps', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    plt.show()
