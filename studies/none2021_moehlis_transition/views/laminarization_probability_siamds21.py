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


#def lam_probability(res, task):
#    with open(os.path.join(res.get_task_path(task), f'trajectories_for_laminarisation_probability_for_true_Moehlis_model_{i}'), 'rb') as f:
#        data = pickle.load(f)
#    task_path = res.get_task_path(task)
#    for energy_i in range(len(data['energy_levels'])):
#        for rp_i in range(len(data['rps'][energy_i])):

#    with open(os.path.join(task_path, 'inputs.json'), 'r') as f:
#        inputs = json.load(f)
#        m = MoehlisFaisstEckhardtModel(Re=inputs['re'], L_x=inputs['l_x'] * np.pi, L_z=inputs['l_z'] * np.pi)

#    filename_and_params = find_all_files_by_named_regexp(r'^(?P<num>\d+)$', task_path)
#    file_paths = [os.path.join(task_path, filename) for filename, params in filename_and_params]
#    p_lams = np.zeros(len(file_paths))
#    for i, file_path in enumerate(file_paths):
#        with open(file_path, 'rb') as f:
#            data_ = pickle.load(f)
#            t = data_['time']
#            ts = data_['timeseries']
#        ke = m.kinetic_energy(ts)
#        N = len(trajs[i])
#        N_lam = 0
#        for ens_member in trajs[i]:
#            ke = m.kinetic_energy(ens_member[:-1, :])
#            if np.all(ke > 10):
#                N_lam += 1
#        p_lams[i] = N_lam / N
##        rt = relaminarisation_time(ke, T=1000, debug=False)
##        relam_times.append(rt if rt is not None else t[-1])
##    for i in range(len(trajs)):
#    return p_lams


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    res_id = 'RC_MOEHLIS'
    res = Research.open(res_id)

    fig, ax = plt.subplots(figsize=(10, 6))
    # (85, 87) for predictions with noise
    # (88,) for predictions without noise
    for tasks, color, label in zip(((83, 86), (85, 87)), ('#ccdeea', 'tab:blue'), ('Truth', 'Prediction')):
        m = MoehlisFaisstEckhardtModel(Re=500, L_x=1.75 * np.pi, L_z=1.2 * np.pi)
        true_lam_probability = []
        trajectory_global_i = 1
        trajectory_numbers = []
        rps = []
        task_inputs = []
        for task in tasks:
            with open(os.path.join(res.get_task_path(task), 'inputs.json'), 'r') as f:
                inputs = json.load(f)
                task_inputs.append(inputs)
        energy_levels = task_inputs[0]['energy_levels']
        for energy_i in range(len(task_inputs[0]['energy_levels'])):
            print(f'Reading energy level #{energy_i}: {task_inputs[0]["energy_levels"][energy_i]}')
            trajs = []
            for task, inputs in zip(tasks, task_inputs):
                for rp_i in range(len(inputs['rps'][energy_i])):
                    with open(os.path.join(res.get_task_path(task), str(inputs['trajectory_numbers'][energy_i][rp_i])), 'rb') as f:
                        data = pickle.load(f)
                        trajs.append(data['timeseries'])
            true_lam_probability.append(laminarization_probability(m, trajs))
            #true_lam_probability.append(trajs)
    #    energy_levels = np.concatenate(energy_levels)
        #true_lam_probability = np.concatenate(true_lam_probability)
        ax.plot(energy_levels, true_lam_probability, 'o--', color=color, markersize=12, label=label)

#        with open(os.path.join(res.local_research_path, 'laminarisation_probability_for_ESN_20_energy_levels_and_50_rps_per_each'), 'rb') as f:
#            data = pickle.load(f)
#        energy_levels = data['energy_levels']
#        predicted_lam_probability = data['lam_probability']
#        ax.plot(energy_levels, predicted_lam_probability, 'o--', color='tab:blue', markersize=12, label='Prediction')


#    m = MoehlisFaisstEckhardtModel(Re=500, L_x=1.75 * np.pi, L_z=1.2 * np.pi)
#    energy_levels = []
#    energy_levels_bckp = []
#    true_lam_probability = []
#    trajectory_numbers = []
#    trajectory_global_i = 1
#    rps_bckp = []
#    for i in [1, 2]:
#        with open(os.path.join(res.get_task_path(83), f'trajectories_for_laminarisation_probability_for_true_Moehlis_model_{i}'), 'rb') as f:
#            data = pickle.load(f)
#        for energy_i in range(len(data['energy_levels'])):
#            trajectory_numbers.append([])
#            energy_levels_bckp.append(data['energy_levels'][energy_i])
#            rps_bckp.append([rp.tolist() for rp in data['rps'][energy_i]])
#            for rp_i in range(len(data['rps'][energy_i])):
#                trajectory_numbers[-1].append(trajectory_global_i)
#                #with open(os.path.join(res.get_task_path(83), str(trajectory_global_i)), 'wb') as f_ts:
#                #    n_steps = len(data['trajectories'][energy_i][rp_i])
#                #    t = np.linspace(0, n_steps, n_steps)
#                #    ts = data['trajectories'][energy_i][rp_i]
#                #    pickle.dump({'time': t, 'timeseries': ts}, f_ts)
#                #trajectory_global_i += 1
#        energy_levels.append(data['energy_levels'])
#        true_lam_probability.append(laminarization_probability(m, data['trajectories']))
#    with open('rps_bckp.json', 'w') as f:
#        json.dump(rps_bckp, f)
#    energy_levels = np.concatenate(energy_levels)
#    true_lam_probability = np.concatenate(true_lam_probability)
#    ax.plot(energy_levels, true_lam_probability, 'o--', color='#ccdeea', markersize=12, label='Truth')


    ax.set_xlabel(r'$E$')
    ax.set_ylabel(r'$P_{lam}(E)$')
    ax.set_xscale('log')
    ax.legend(fontsize=16)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax.grid()
    plt.tight_layout()
    plt.savefig('p_lam_esn.eps', dpi=200)
    plt.show()
