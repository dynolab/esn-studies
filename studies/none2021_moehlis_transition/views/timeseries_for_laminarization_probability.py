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
from comsdk.misc import load_from_json, find_all_files_by_named_regexp
from thequickmath.reduced_models.transition_to_turbulence import MoehlisFaisstEckhardtModel


def plot_trajectories(ax, data_path):
    filename_and_params = find_all_files_by_named_regexp(r'^(?P<num>\d+)$', data_path)
    file_paths = [os.path.join(data_path, filename) for filename, params in filename_and_params]
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            data_ = pickle.load(f)
            t = data_['time']
            ts = data_['timeseries']
        ke = m.kinetic_energy(ts)
        ax.plot(t, ke, linewidth=1)
        #ax.set_xlim((-100, 7000))
        #ax.set_ylim((0, 22))
    ax.grid()


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    res_id = 'RC_MOEHLIS'
    res = Research.open(res_id)
    data_paths_truth = [res.get_task_path(t) for t in [83]]  # Moehlis model, truth
    # data_paths_pred = [res.get_task_path(t) for t in [85]]  # ESN, prediction
    data_paths_pred = [os.path.join(res.get_task_path(t), 'esn_re_500_trained_wo_lam_event_2') for t in [113]]  # ESN ensemble, prediction
    with open(os.path.join(data_paths_truth[0], 'inputs.json'), 'r') as f:
        inputs = json.load(f)
        m = MoehlisFaisstEckhardtModel(Re=inputs['re'], L_x=inputs['l_x'] * np.pi, L_z=inputs['l_z'] * np.pi)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for p_truth, p_pred in zip(data_paths_truth, data_paths_pred):
        plot_trajectories(axes[0], p_truth)
        plot_trajectories(axes[1], p_pred)
    axes[0].set_title('Truth', fontsize=16)
    axes[1].set_title('Prediction', fontsize=16)
    plt.tight_layout()
    #plt.savefig('moehlis_350_trajs.png', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    plt.show()
