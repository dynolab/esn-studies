import os
import sys
sys.path.append(os.getcwd())
import pickle
import json
from operator import itemgetter

import numpy as np
from scipy.stats import expon
import matplotlib
import matplotlib.pyplot as plt

import restools
from studies.none2021_predicting_transition_using_reservoir_computing.extensions import relaminarisation_time, survival_function
from comsdk.research import Research
from comsdk.misc import load_from_json, find_all_files_by_named_regexp
from thequickmath.reduced_models.transition_to_turbulence import MoehlisFaisstEckhardtModel


def build_relaminarisation_times_from_tasks(res, tasks):
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
            rt = relaminarisation_time(ke, T=1000, debug=False)
            relam_times.append(rt if rt is not None else t[-1])
    return relam_times


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    res_id = 'RC_MOEHLIS'
    res = Research.open(res_id)
#    data = {
#        'Truth': {
#            're_values': [250, 275, 300, 325, 350, 200],
#            'tasks': [
#                [11, 14],
#                [42, 45],
#                list(range(1, 11)) + [15],
#                [34, 35],
#                [53],
#                [12, 13],
#            ],
#        },
#        'ESN': {
#            're_values': [250, 275, 300, 325, 350],
#            'tasks': [
#                [29, 30],
#                [51],
#                list(range(17, 28)),
#                [44],
#                [55],
#            ],
#        }
#    }

    data = {
        'Truth': {
            're_values': [200, 250, 275, 300, 350],
            'tasks': [
                [12, 13],
                [11, 14],
                [42, 45],
                list(range(1, 11)) + [15],
                [53],
            ],
        },
        'ESN': {
            're_values': [250, 275, 300],
            'tasks': [
                [29, 30],
                [51, 52],  # this is with noise
                #[89, 90],  # this is without noise
                list(range(17, 28)),
            ],
        }
    }


    colors = {
        'light': [
            '#ccdeea',
            '#ffd8b6',
            '#b6deb6',
            '#e7aeaf',
            '#cab7dc',
            '#d0b1ab',
            #'tab:blue', #  #1f77b4  (light version is #ccdeea)
            #'tab:orange', #  #ff7f0e (light version is #ffd8b6)
            #'tab:green', #  #2ca02c (light version is #b6deb6)
            #'tab:red', #  #d62728 (light version is #e7aeaf)
            #'tab:purple', #9467bd (light version is )
            #(0.7, 0.6, 0.6),  # 'tab:blue'
            #(0.6, 0.7, 0.6),  # 'tab:orange'
            #(0.6, 0.6, 0.7),
            #(0.7, 0.6, 0.7),
        ],
        'dark': [
            'tab:blue',
            'tab:orange',
            'tab:green',
            'tab:red',
            'tab:purple',
            'tab:brown',
            #(0.7, 0.0, 0.0),  #
            #(0.0, 0.7, 0.0),
            #(0.0, 0.0, 0.7),
            #(0.7, 0.0, 0.7),
        ],
    }


#'tab:blue'
#'tab:orange'
#'tab:green'
#'tab:red'
#'tab:purple'
#'tab:brown'
#'tab:pink'
#'tab:gray'
#'tab:olive'
#'tab:cyan'
    expon_law_fits = {'Truth': {}, 'ESN': {}}

    lines_for_legend = []
    re_for_legend = []

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    for re_i in range(len(data['Truth']['re_values'])):
        re = data['Truth']['re_values'][re_i]
        relam_times_truth = build_relaminarisation_times_from_tasks(res, data['Truth']['tasks'][re_i])
        expon_law_t_0, expon_law_tau = expon.fit(np.array(relam_times_truth, dtype=float))
        expon_law_fits['Truth'][re] = {'t_0': expon_law_t_0, 'tau': expon_law_tau}
        lines = ax.semilogy(*survival_function(relam_times_truth), 'o--', color=colors['light'][re_i], linewidth=2, markersize=10)
        if re not in data['ESN']['re_values']:
            #  r'$Re = ' + str(re) + r'$'
            lines_for_legend.append(lines[0])
            re_for_legend.append(re)
        #t = np.linspace(0, ax.get_xlim()[1] * 0.75, 200)
        #lines = ax.semilogy(t, expon(expon_law_t_0, expon_law_tau).sf(t))

    for re_i in range(len(data['Truth']['re_values'])):
        re = data['Truth']['re_values'][re_i]
        if re in data['ESN']['re_values']:
            relam_times_esn = build_relaminarisation_times_from_tasks(res, data['ESN']['tasks'][re_i - 1])
            expon_law_t_0, expon_law_tau = expon.fit(np.array(relam_times_esn, dtype=float))
            expon_law_fits['ESN'][re] = {'t_0': expon_law_t_0, 'tau': expon_law_tau}
            lines = ax.semilogy(*survival_function(relam_times_esn), '^--', color=colors['dark'][re_i], linewidth=2, markersize=10)
            lines_for_legend.append(lines[0])
            re_for_legend.append(re)

    for re in expon_law_fits['ESN'].keys():
        print(f'Re = {re}, Truth exponential law parameters: t_0 = {expon_law_fits["Truth"][re]["t_0"]}, '
              f'tau = {expon_law_fits["Truth"][re]["tau"]}')
        print(f'Re = {re}, ESN exponential law parameters: t_0 = {expon_law_fits["ESN"][re]["t_0"]}, '
              f'tau = {expon_law_fits["ESN"][re]["tau"]}, '
              f'Relative error in tau: {np.abs(expon_law_fits["ESN"][re]["tau"] - expon_law_fits["Truth"][re]["tau"]) / np.abs(expon_law_fits["Truth"][re]["tau"])}')
        expon_law_fits['ESN'][re]
    ax.grid()
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$S(t)$')
    ax.set_xlim((-100, 16000))
    lines_and_re_sorted = sorted(zip(lines_for_legend, re_for_legend), key=itemgetter(1))
    lines_for_legend = [pair[0] for pair in lines_and_re_sorted]
    labes_for_legend = [r'$Re = ' + str(pair[1]) + r'$' for pair in lines_and_re_sorted]
    ax.legend(lines_for_legend, labes_for_legend, fontsize=16)
    plt.tight_layout()
    plt.savefig('lifetime_distrs.eps', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    plt.show()
