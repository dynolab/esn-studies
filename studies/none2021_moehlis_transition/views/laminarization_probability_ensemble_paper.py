import os
import sys
sys.path.append(os.getcwd())
import pickle
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import restools
from studies.none2021_moehlis_transition.extensions import relaminarisation_time, \
    survival_function, laminarization_probability
from comsdk.research import Research
from comsdk.misc import load_from_json, find_all_files_by_named_regexp
from thequickmath.reduced_models.transition_to_turbulence import MoehlisFaisstEckhardtModel
from thequickmath.stats import EmpiricalDistribution


class LamProbabilitySummary:
    def __init__(self, num):
        self.medians = np.zeros((num,))
        self.ppf01 = np.zeros_like(self.medians)
        self.ppf09 = np.zeros_like(self.medians)


def p_lam(data_paths, task_inputs):
    energy_levels = task_inputs[0]['energy_levels']
    lam_probability = []
    for energy_i in range(len(task_inputs[0]['energy_levels'])):
        print(f'Reading energy level #{energy_i}: {task_inputs[0]["energy_levels"][energy_i]}')
        trajs = []
        for dp, inputs in zip(data_paths, task_inputs):
            for rp_i in range(len(inputs['rps'][energy_i])):
                with open(os.path.join(dp, str(inputs['trajectory_numbers'][energy_i][rp_i])), 'rb') as f:
                    data = pickle.load(f)
                    trajs.append(data['timeseries'])
        lam_probability.append(laminarization_probability(m, trajs))
    return energy_levels, lam_probability


def build_task_inputs(data_paths, inputs_filename):
    task_inputs = []
    for dp in data_paths:
        with open(os.path.join(dp, inputs_filename), 'r') as f:
            inputs = json.load(f)
            task_inputs.append(inputs)
    return task_inputs


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    res_id = 'RC_MOEHLIS'
    res = Research.open(res_id)

    fig, ax = plt.subplots(figsize=(6, 5))
    # (85, 87) for predictions with noise
    # (88,) for predictions without noise
    data_paths_truth = [res.get_task_path(t) for t in [83, 86]]  # Moehlis model, truth
    # data_paths_pred = [res.get_task_path(t) for t in [85]]  # ESN, prediction
    data_high_level_paths_pred = [
        [113, 123],  # random seeds from 1 to 10
        [114, 124],  # random seeds from 11 to 20
        [115, 125],  # random seeds from 21 to 30
        [116, 126],  # random seeds from 31 to 40
        [117, 127],  # random seeds from 41 to 50
        [118, 128],  # random seeds from 51 to 60
        [119, 129],  # random seeds from 61 to 70
        [120, 130],  # random seeds from 71 to 80
        [121, 131],  # random seeds from 81 to 90
        [122, 132],  # random seeds from 91 to 100
    ]
    #data_paths_pred = [os.path.join(res.get_task_path(t), 'esn_re_500_trained_wo_lam_event_1') for t in [116]]  # ESN ensemble, prediction

    #with open(os.path.join(data_paths_truth[0], 'inputs.json'), 'r') as f:
    #    truth_inputs = json.load(f)
    #for i in range(1, 11):
    #    pred_inputs_path = os.path.join(res.get_task_path(113), f'ti_inputs_{i}.json')
    #    with open(pred_inputs_path, 'r') as f:
    #        pred_inputs = json.load(f)
    #    pred_inputs['trajectory_numbers'] = truth_inputs['trajectory_numbers']
    #    pred_inputs['energy_levels'] = truth_inputs['energy_levels']
    #    pred_inputs['rps'] = truth_inputs['rps']
    #    with open(pred_inputs_path, 'w') as f:
    #        json.dump(pred_inputs, f)
    m = MoehlisFaisstEckhardtModel(Re=500, L_x=1.75 * np.pi, L_z=1.2 * np.pi)
    # Build true curve
    task_inputs = build_task_inputs(data_paths_truth, 'inputs.json')
    true_energy_levels, true_lam_probability = p_lam(data_paths_truth, task_inputs)
    ax.plot(true_energy_levels, true_lam_probability,
            'o--',
            color='#a0ccea',  # ccdeea
            markersize=12,
            label='Truth')
    # Build ensemble distribution
    lam_probability_array_list = []
    for data_high_level_paths in data_high_level_paths_pred:
        for i in range(1, 11):
            data_paths = [os.path.join(res.get_task_path(t), f'esn_re_500_trained_wo_lam_event_{i}') for t in data_high_level_paths]
            task_inputs = build_task_inputs(data_paths, '../ti_inputs_1.json')
            energy_levels, lam_probability = p_lam(data_paths, task_inputs)
            lam_probability_array_list.append(lam_probability)
    lam_probability_array_list = np.array(lam_probability_array_list)
    el_num = lam_probability_array_list.shape[1]
    p_lam_summary = LamProbabilitySummary(el_num)
    for e_i in range(el_num):
        distr = EmpiricalDistribution(lam_probability_array_list[:, e_i])
        p_lam_summary.medians[e_i] = distr.median()
        p_lam_summary.ppf01[e_i] = distr.ppf(0.1)
        p_lam_summary.ppf09[e_i] = distr.ppf(0.9)
    #mean_lam_probability = lam_probability_array_list.mean(axis=0)
    ax.plot(energy_levels, p_lam_summary.medians,
            'o--',
            color='tab:blue',
            markersize=12,
            label='Prediction')
    ax.set_xlabel(r'$E$')
    ax.set_ylabel(r'$P_{lam}(E)$')
    ax.set_xscale('log')
    ax.fill_between(
        energy_levels,
        p_lam_summary.ppf01,
        p_lam_summary.ppf09,
        alpha=0.1,
        color='tab:blue',
        linewidth=2,
        zorder=-10)
    ax.legend(fontsize=14)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    #ax.grid()
    ax.set_rasterization_zorder(0)
    plt.tight_layout()
    plt.savefig('p_lam.eps', dpi=200)
    plt.show()
