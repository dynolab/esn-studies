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

    task = 92
    res_id = 'RC_MOEHLIS'
    res = Research.open(res_id)
    task_path = res.get_task_path(task)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for i in range(1, 11):
        with open(os.path.join(task_path, 'esn_re_275_4', str(i)), 'rb') as f:
        #with open(os.path.join(res.local_research_path, 'training_timeseries_re_275_0_3100'), 'rb') as f:
        #with open(os.path.join(res.local_research_path, 'training_timeseries_re_275_new'), 'rb') as f:
        #with open(os.path.join(task_path, 'training_timeseries_re_275'), 'rb') as f:
            data = pickle.load(f)
            t = data['time']
            ts = data['timeseries']
#        if t[-1] > 6000:
#            print(t[-1], i)
        # 34 -> 27
#        if i != 9:  # 11 for 42
#            continue
        with open(os.path.join(task_path, 'ti_inputs_1.json'), 'r') as f:
            inputs = json.load(f)
            m = MoehlisFaisstEckhardtModel(Re=inputs['re'], L_x=inputs['l_x'] * np.pi, L_z=inputs['l_z'] * np.pi)
            #m = MoehlisFaisstEckhardtModel(Re=inputs['original_model_parameters']['re'], L_x=inputs['original_model_parameters']['l_x'] * np.pi, L_z=inputs['original_model_parameters']['l_z'] * np.pi)
        #ts -= m.laminar_state
        #with open(os.path.join(res.local_research_path, 'training_timeseries_re_275_new_pert'), 'wb') as f:
#            d = {
#                'time': data['time'][data['time'] < 5500],
#                'timeseries': data['timeseries'][data['time'] < 5500]
#            }
        #    d = {
        #        'time': t,
        #        'timeseries': ts
        #    }
        #    pickle.dump(d, f)
        ke = m.kinetic_energy(ts)
        ax.plot(t, ke, linewidth=2)
        #ax.set_xlim((-100, 7000))
        #ax.set_ylim((0, 22))
    ax.grid()
    plt.tight_layout()
    #plt.savefig('moehlis_350_trajs.png', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    plt.show()
