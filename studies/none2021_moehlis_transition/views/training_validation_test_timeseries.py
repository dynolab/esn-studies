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
from restools.plotting import rasterise_and_save, reduce_eps_size


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    task = 2
    res_id = 'RC_MOEHLIS'
    res = Research.open(res_id)
    task_path = res.get_task_path(task)

    with open(os.path.join(res.local_research_path, 'new_ts.json'), 'r') as f:
        d = json.load(f)
    data = np.array(d['timeseries'])[::10, :]
    trainlen = 15000
    synclen = 10
    predlen = 300

    testlen = synclen + predlen
    test_timeseries_set = [data[i*testlen: (i+1)*testlen] for i in range(3)]
    training_timeseries = data[4*testlen:4*testlen + trainlen]
    training_timeseries_wo_lam = training_timeseries[:-2500]
    m = MoehlisFaisstEckhardtModel(Re=500, L_x=1.75 * np.pi, L_z=1.2 * np.pi)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    test_timeseries = np.r_[test_timeseries_set[0], test_timeseries_set[1], test_timeseries_set[2]]
    ke = m.kinetic_energy(test_timeseries)
    ax.plot(range(len(ke)), ke, color='gray', linewidth=3)
    ax.set_xlabel(r'$t$', fontsize=16)
    ax.set_ylabel(r'$E$', fontsize=16)
    ax.grid()
    obj_to_rasterize = []
    obj = ax.fill_between(range(300), 0.5, 4, alpha=0.2)
    obj_to_rasterize.append(obj)
    obj = ax.fill_between(range(300, 600), 0.5, 4, alpha=0.2)
    obj_to_rasterize.append(obj)
    obj = ax.fill_between(range(600, 900), 0.5, 4, alpha=0.2)
    obj_to_rasterize.append(obj)
    plt.tight_layout()
    fname = 'test_timeseries.eps'
    rasterise_and_save(fname, rasterise_list=obj_to_rasterize, fig=fig, dpi=300)
    reduce_eps_size(fname)
    plt.show()

#with open('/media/tony/Seagate/Leeds/PhD/research/2021-04-30_Predicting_transition_to_turbulence_using_reservoir_computing/new_ts.json', 'r') as f:
#    d = json.load(f)
#data = np.array(d['timeseries'])[::10, :]
#trainlen = 15000
#synclen = 10
#predlen = 300
#
#testlen = synclen + predlen
#test_timeseries_set = [data[i*testlen: (i+1)*testlen] for i in range(3)]
#training_timeseries = data[4*testlen:4*testlen + trainlen]
#training_timeseries_wo_lam = training_timeseries[:-2500]
#
#m = MoehlisFaisstEckhardtModel(Re=500, L_x=1.75 * np.pi, L_z=1.2 * np.pi)
#
#with open('/media/tony/Seagate/Leeds/PhD/research/2021-04-30_Predicting_transition_to_turbulence_using_reservoir_computing/moehlis_dataset_Re_500.json', 'r') as f:
#    d = json.load(f)
#timeseries_out_of_training = np.array(d['timeseries'])[::10, :]
#
#import matplotlib
#
#matplotlib.rcParams['figure.dpi'] = 200
#
#fig, axes = plt.subplots(3, 1, figsize=(13, 7))
#test_timeseries = np.r_[test_timeseries_set[0], test_timeseries_set[1], test_timeseries_set[2]]
#for ax, ts, title in zip(axes, (training_timeseries, test_timeseries, timeseries_out_of_training), ('Training time series', 'Validation time series', 'Test time series')):
#    ke = m.kinetic_energy(ts)
#    ax.plot(range(len(ke)), ke, color='gray', linewidth=3)
#    ax.set_ylabel(r'$E$', fontsize=16)
#    ax.set_title(title, fontsize=18)
#    ax.grid()
#axes[1].fill_between(range(300), 0.5, 4, alpha=0.2)
#axes[1].fill_between(range(300, 600), 0.5, 4, alpha=0.2)
#axes[1].fill_between(range(600, 900), 0.5, 4, alpha=0.2)
#axes[-1].set_xlabel('$t$', fontsize=16)
#plt.tight_layout()
##plt.savefig('long_term_prediction_on_training_set.png', dpi=200)
#plt.show()