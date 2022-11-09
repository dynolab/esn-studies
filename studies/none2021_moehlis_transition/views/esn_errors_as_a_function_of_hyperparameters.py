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
from studies.none2021_moehlis_transition.extensions import relaminarisation_time, survival_function
from comsdk.research import Research
from comsdk.misc import load_from_json, find_all_files_by_named_regexp
from thequickmath.reduced_models.transition_to_turbulence import MoehlisFaisstEckhardtModel


class HyperparameterSearchSummary:
    def __init__(self, n_rho, n_s):
        self.mean = np.zeros((n_rho, n_s))
        self.max = np.zeros_like(self.mean)
        self.min = 10**10 * np.ones_like(self.mean)


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    res_id = 'RC_MOEHLIS'
    res = Research.open(res_id)
    tasks = [
        104,
        105,
        106,
    ]
    with open(os.path.join(res.get_task_path(tasks[0]), 'inputs.json'), 'r') as f:
        inputs = json.load(f)
    hss = HyperparameterSearchSummary(
        n_rho=len(inputs['spectral_radius_values']),
        n_s=len(inputs['sparsity_values']),
    )
    for task in tasks:
        with open(os.path.join(res.get_task_path(task), 'esn_errors.obj'), 'rb') as f:
            esn_errors = pickle.load(f)
        average_along_trials = np.mean(esn_errors, axis=2)
        hss.mean += average_along_trials
        hss.max = np.maximum(hss.max, average_along_trials)
        hss.min = np.minimum(hss.min, average_along_trials)
    hss.mean /= len(tasks)
    X, Y = np.meshgrid(inputs['spectral_radius_values'], inputs['sparsity_values'], indexing='ij')
    # inputs['spectral_radius_values']
    # inputs['sparsity_values']
    # inputs['trial_number']
    print('qwer')

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for rho_i in range(esn_errors.shape[0]):
        ax.errorbar(
            inputs['sparsity_values'],
            hss.mean[rho_i, :],
            yerr=np.stack((hss.mean[rho_i, :] - hss.min[rho_i, :], hss.max[rho_i, :] - hss.mean[rho_i, :]), axis=0),
            fmt='o--',
            capsize=4,
            label=r'$\rho = ' + str(inputs['spectral_radius_values'][rho_i]) + r'$'
        )
    ax.grid()
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$E$')
    #ax.set_xlim((-100, 16000))
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('esn_errors_as_a_function_of_hyperparameters.eps', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    plt.show()
