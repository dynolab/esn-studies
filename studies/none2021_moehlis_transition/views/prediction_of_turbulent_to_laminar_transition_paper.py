import os
import sys
sys.path.append(os.getcwd())
import pickle
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import restools
from studies.none2021_moehlis_transition.extensions import is_laminarised
from comsdk.research import Research
from comsdk.misc import load_from_json, find_all_files_by_named_regexp
from thequickmath.reduced_models.transition_to_turbulence import MoehlisFaisstEckhardtModel


def compute_probability_of_turbulent_to_laminar_transition(res, task, model):
    task_path = res.get_task_path(task)
    lam_fraction = 0.
    filename_and_params = find_all_files_by_named_regexp(r'^(?P<num>\d+)$', task_path)
    file_paths = [os.path.join(task_path, filename) for filename, params in filename_and_params]
    n_ensemble_members = len(file_paths)
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            data_ = pickle.load(f)
            t = data_['time']
            ts = data_['timeseries']
        ke = model.kinetic_energy(ts)
        if is_laminarised(ke, T=500, debug=False):
            lam_fraction += 1
    lam_fraction /= n_ensemble_members
    return lam_fraction


def plot_trajectories(axes, res, trajs, time_shift=0, color=None, linewidth=3, plot_point_at=None):
    for ax, traj_data in zip(axes, trajs):
        task_path = res.get_task_path(traj_data['task'])
        point_t = 0
        point_ts = 0
        with open(os.path.join(task_path, str(traj_data['i'])), 'rb') as f:
            data = pickle.load(f)
            t = data['time'][traj_data['begin_time']:traj_data['end_time']] + time_shift
            ts = data['timeseries'][traj_data['begin_time']:traj_data['end_time']]
            # print(310*4+15000-2300)
            if plot_point_at is not None:
                point_t = data['time'][plot_point_at]
                point_ts = m.kinetic_energy(data['timeseries'][plot_point_at])
        ke = m.kinetic_energy(ts)
        ax.plot(t, ke, linewidth=linewidth, color=color)
        if plot_point_at is not None:
            ax.plot([point_t], [point_ts], 'ro', markersize=10)


def plot_trajectories_at_one_plot(ax, res, trajs, time_shift=0, color=None, linewidth=3, plot_point_at=None):
    for traj_data in trajs:
        task_path = res.get_task_path(traj_data['task'])
        point_t = 0
        point_ts = 0
        with open(os.path.join(task_path, str(traj_data['i'])), 'rb') as f:
            data = pickle.load(f)
            t = data['time'][traj_data['begin_time']:traj_data['end_time']] + time_shift
            ts = data['timeseries'][traj_data['begin_time']:traj_data['end_time']]
            # print(310*4+15000-2300)
            if plot_point_at is not None:
                point_t = data['time'][plot_point_at] + time_shift
                point_ts = m.kinetic_energy(data['timeseries'][plot_point_at])
        ke = m.kinetic_energy(ts)
        ax.plot(t, ke, linewidth=linewidth, color=color)
        if plot_point_at is not None:
            ax.plot([point_t], [point_ts], 'ro', markersize=10)


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    res_id = 'RC_MOEHLIS'
    res = Research.open(res_id)
    begin_time = 12830
    time_shift = 13940
    delta_time = 100
    tasks_esn = [91, 38, 39, 40, 41]
    trajs_esn_all = [
        [{'task': task, 'i': i, 'begin_time': 0, 'end_time': -1} for i in range(1, 11)]
            for task in tasks_esn]
    trajs_moehlis = [
        {'task': 37, 'i': 1, 'begin_time': begin_time, 'end_time': 16000},
        {'task': 37, 'i': 1, 'begin_time': begin_time, 'end_time': 16000},
        {'task': 37, 'i': 1, 'begin_time': begin_time, 'end_time': 16000},
        {'task': 37, 'i': 1, 'begin_time': begin_time, 'end_time': 16000},
        {'task': 37, 'i': 1, 'begin_time': begin_time, 'end_time': 16000},
    ]


    #  Initial times: T = 13940 - 100, 13940, 13940 + 100, 13940 + 200, 13940 + 300
    task_path = res.get_task_path(trajs_esn_all[0][0]['task'])
    #ens_members = [9, 10, 7, 8]  # 17, 4 = 1500,      18, 8 = 4000,     19, 5 = 2800      19, 10 = 5400
    with open(os.path.join(task_path, 'inputs.json'), 'r') as f:
        inputs = json.load(f)
        m = MoehlisFaisstEckhardtModel(Re=inputs['re'], L_x=inputs['l_x'] * np.pi, L_z=inputs['l_z'] * np.pi)
    fig, axes = plt.subplots(len(tasks_esn), 1, figsize=(6, 7), sharex=True, sharey=True)
    plot_trajectories(axes, res, trajs_moehlis, color='#ccdeea', linewidth=3)
    probs = []
    for i in range(len(trajs_esn_all)):
        probs.append(compute_probability_of_turbulent_to_laminar_transition(res, tasks_esn[i], m))
        plot_trajectories_at_one_plot(axes[i], res, trajs_esn_all[i], time_shift=time_shift + (i - 1)*delta_time,
                                      plot_point_at=0,
                                      linewidth=2,
                                      color='tab:blue')
    t_lims = axes[0].get_xlim()
    lam_state_ke = m.kinetic_energy(m.laminar_state)
    for ax, p in zip(axes, probs):
        ax.text(12800, 20, r'$P_{\text{T} \to \text{L}} = ' + str(p) + r'$', fontsize=16,
                bbox=dict(boxstyle=matplotlib.patches.BoxStyle('Square', pad=0.5),
                   ec=(0.0, 0.0, 0.0),
                   fc=(1.0, 1.0, 1.0),
                   ))
        #ax.plot([begin_time, t_lims[1]], [lam_state_ke, lam_state_ke], 'k--', linewidth=3)
        ax.set_ylabel(r'$E$')
        ax.set_ylim((-2, 35))
        ax.grid()
    axes[-1].set_xlabel(r'$t$')
    plt.tight_layout()
    plt.savefig('prediction_of_turbulent_to_laminar_transition.eps', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    plt.show()
