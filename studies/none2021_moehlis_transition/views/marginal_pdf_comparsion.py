import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from restools.timeintegration import TimeIntegrationLowDimensional
from comsdk.research import Research
from comsdk.misc import find_all_files_by_named_regexp
from thequickmath.reduced_models.transition_to_turbulence import MoehlisFaisstEckhardtModel


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    res_id = 'RC_MOEHLIS'
    res = Research.open(res_id)
    data = {
        'moehlis_tasks': [16],
        'esn_tasks': [1]
    }
    moehlis_concated_timeseries = []
    for task in data['moehlis_tasks']:
        task_path = res.get_task_path(task)
        filename_and_params = find_all_files_by_named_regexp(r'^(?P<num>\d+)$', task_path)
        file_paths = [os.path.join(task_path, filename) for filename, params in filename_and_params]
        indices = [filename for filename, _ in filename_and_params]
        for i in indices:
            ti = TimeIntegrationLowDimensional(task_path, i)
            moehlis_concated_timeseries.append(ti.timeseries)
    moehlis_concated_timeseries = np.concatenate(moehlis_concated_timeseries)

    # PLOT MARGINAL PDF COMPARISON
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    for row in range(3):
        for col in range(3):
            i = 3*row + col
            sns.kdeplot(data=moehlis_concated_timeseries[:, i],
                        ax=axes[row][col], label='Truth')
            axes[row][col].set_xlabel(f'$u_{i+1}$', fontsize=16)
            axes[row][col].legend()
            axes[row][col].grid()
    plt.tight_layout(pad=1)
    #plt.savefig('phase_space.png', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    plt.show()

    # PLOT TWO_DIMENSIONAL JOINT PDF COMPARISON
    #print(moehlis_concated_timeseries.shape)
    #moehlis_concated_timeseries = moehlis_concated_timeseries[::50]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    n = 0
    for i in range(3):
        for j in range(i + 1, 4):
            row = int(n//3)
            col = n - int(n//3) * 3
            #sns.kdeplot(x=prediction[:, i+1], y=best_prediction[:, j+1],
            #            ax=axes[row][col], label='Prediction')
            sns.histplot(x=moehlis_concated_timeseries[:, i+1], y=moehlis_concated_timeseries[:, j+1],
                         ax=axes[row][col], label='Truth', bins=50, stat='density')
#            sns.kdeplot(x=moehlis_concated_timeseries[:, i+1], y=moehlis_concated_timeseries[:, j+1],
#                         ax=axes[row][col], label='Truth', levels=[0.2, 0.4, 0.6, 0.8])
            axes[row][col].set_xlabel(f'$u_{i+1}$', fontsize=16)
            axes[row][col].set_ylabel(f'$u_{j+1}$', fontsize=16)
            axes[row][col].legend()
            axes[row][col].grid()
            n += 1
    plt.tight_layout(pad=1)
    #plt.savefig('phase_space.png', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    plt.show()
