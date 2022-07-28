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

    res_id = 'RC_MOEHLIS'
    res = Research.open(res_id)
#    ts_datum = [
#        {'re': 250, 'ts_name': 'training_timeseries_re_250', 'meta_data_task': 30},
#        {'re': 275, 'ts_name': 'training_timeseries_re_275', 'meta_data_task': 47},
#        {'re': 300, 'ts_name': 'training_timeseries_re_300', 'meta_data_task': 17},
##        {'re': 500, 'ts_name': 'training_timeseries_re_500', 'meta_data_task': 19},
#        {'re': 500, 'ts_name': 'training_timeseries_re_500_wo_lam_event', 'meta_data_task': 19},
#    ]

    ts_datum = [
        {
            're': 250,
            'ts_name': 'training_timeseries_re_250',
            'original_task': 11,
            'original_timeseries': '21',
            'original_training_begin_time': 0,
            'original_training_end_time': 6000,
            'original_laminarization_event_end_time': -1,
            'meta_data_task': 30,
            'text_location': 'right',
        },
        {
            're': 275,
            'ts_name': 'training_timeseries_re_275',
            'original_task': 42,
            'original_timeseries': '9',
            'original_training_begin_time': 0,
            'original_training_end_time': 5700,
            'original_laminarization_event_end_time': -1,
            'meta_data_task': 51,
            'text_location': 'right',
        },
        {
            're': 300,
            'ts_name': 'training_timeseries_re_300',
            'original_task': 82,
            'original_timeseries': '1',
            'original_training_begin_time': 310*4,
            'original_training_end_time': -1,
            'original_laminarization_event_end_time': -1,
            'meta_data_task': 17,
            'text_location': 'left',
        },
#        {'re': 500, 'ts_name': 'training_timeseries_re_500', 'meta_data_task': 19},
        {
            're': 500,
            'ts_name': 'training_timeseries_re_500_wo_lam_event',
            'original_task': 37,
            'original_timeseries': '1',
            'original_training_begin_time': 310*4,
            'original_training_end_time': 310*4 + 15000 - 2500,
            'original_laminarization_event_end_time': 310*4 + 15000,
            'meta_data_task': 19,
            'text_location': 'left',
        },
    ]

#    #with open(os.path.join(res.get_task_path(37), '1'), 'rb') as f:
#    with open(os.path.join(res.local_research_path, 'raw_timeseries_re_300'), 'rb') as f:
#        data = pickle.load(f)
#        t = data['time']
#        ts = data['timeseries']
##        trainlen = 15000
#        synclen = 10
#        predlen = 300
#        testlen = synclen + predlen
#        test_ts = ts[:4*testlen]
#        test_t = t[:4*testlen]
#        training_ts = ts[4*testlen:]
#        training_t = t[4*testlen:] - t[4*testlen]
##        training_ts = ts[4*testlen:4*testlen + trainlen]
##        training_t = t[4*testlen:4*testlen + trainlen] - t[4*testlen]
##        training_ts_wo_lam = training_ts[:-2500]
##        training_t_wo_lam = training_t[:-2500] - training_t[0]
#        with open(os.path.join(res.local_research_path, 'training_timeseries_re_300'), 'wb') as f_out:
#            pickle.dump({'time': training_t, 'timeseries': training_ts}, f_out)
##        with open(os.path.join(res.local_research_path, 'training_timeseries_re_500_wo_lam_event'), 'wb') as f_out:
##            pickle.dump({'time': training_t_wo_lam, 'timeseries': training_ts_wo_lam}, f_out)
#        with open(os.path.join(res.local_research_path, 'test_timeseries_re_300'), 'wb') as f_out:
#            pickle.dump({'time': test_t, 'timeseries': test_ts}, f_out)

    fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True)
    for ax, ts_data in zip(axes, ts_datum):
        task_path = res.get_task_path(ts_data['meta_data_task'])
        with open(os.path.join(task_path, 'inputs.json'), 'r') as f:
            inputs = json.load(f)
            m = MoehlisFaisstEckhardtModel(Re=inputs['re'], L_x=inputs['l_x'] * np.pi, L_z=inputs['l_z'] * np.pi)
#        with open(os.path.join(res.local_research_path, ts_data['ts_name']), 'rb') as f:
#            data = pickle.load(f)
#            t = data['time']
#            ts = data['timeseries']
        with open(os.path.join(res.get_task_path(ts_data['original_task']), ts_data['original_timeseries']), 'rb') as f:
            data = pickle.load(f)
        ke_last = None
        t_last = None
#        for start_i, end_i, time_shift, color in zip((ts_data['original_training_begin_time'], ts_data['original_training_end_time']),
#                                  (ts_data['original_training_end_time'], ts_data['original_laminarization_event_end_time']),
#                                  (data['time'][ts_data['original_training_begin_time']], data['time'][ts_data['original_training_begin_time']]),
#                                  ('tab:blue', '#ccdeea')):
#            t = data['time'][start_i:end_i] - time_shift
#            ts = data['timeseries'][start_i:end_i]
#            ke = m.kinetic_energy(ts)
#            ax.plot(t, ke, linewidth=2, color=color)
#            ke_last = ke
#            t_last = t

        for start_i, end_i, time_shift, mode in zip((ts_data['original_training_begin_time'], ts_data['original_training_end_time']),
                                  (ts_data['original_training_end_time'], ts_data['original_laminarization_event_end_time']),
                                  (data['time'][ts_data['original_training_begin_time']], data['time'][ts_data['original_training_begin_time']]),
                                  ('curve+shadow', 'curve')):
                                  #('tab:blue', '#ccdeea')):
            t = data['time'][start_i:end_i] - time_shift
            ts = data['timeseries'][start_i:end_i]
            ke = m.kinetic_energy(ts)
            if mode == 'curve+shadow':
                ax.fill_between([t[0], t[-1]], 0, 27, color='#ccc')
            ax.plot(t, ke, linewidth=3, color='tab:blue')
            ke_last = ke
            t_last = t

        if len(t_last) > 0:
            ax.plot([t_last[-1] + t_last[-1] - t_last[-2], 15000], [ke_last[-1], ke_last[-1]], linewidth=3, color='tab:blue')
#        print(len(t))
        ax.set_yticks((0, 10, 20))
        ax.set_xlim((-10, 15700))
        ax.set_ylim((0, 27))
        ax.set_ylabel(r'$E$')
        ax.set_title(r'$Re = ' + str(ts_data['re']) + r'$')
        ax.grid()
    axes[-1].set_xlabel(r'$t$')
    plt.tight_layout(h_pad=0.2)
    plt.savefig('all_training_timeseries.eps', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    plt.show()
