import os
import sys
sys.path.append(os.getcwd())

import restools
from comsdk.research import Research

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

from pyESN import ESN, optimal_esn
from skesn.esn import EsnForecaster


if __name__ == '__main__':
    res = Research.open('RC_MOEHLIS')
    task_num = 42

    data = []
    time = []
    n_files = 100

    for num_data in range(n_files):
        with open(os.path.join(res.get_task_path(task_num), str(num_data+1)), 'rb') as a:
            b = pickle.load(a)
        data.append(b['timeseries'])
        time.append(b['time'])

    df_time_len = pd.DataFrame(
        {
            'time_len': [len(i) for i in time],
            'file_name': [i+1 for i in range(n_files)]
        })

    df_time_len.sort_values(by=['time_len'], inplace=True)

    plt.figure(figsize=(25,6))
    plt.plot(range(n_files), df_time_len.time_len, '.-')
    for i, j, k in zip(df_time_len.time_len, df_time_len.file_name, range(n_files)):
        plt.text(k, i+100, j, fontsize=8.)
    plt.grid()
    plt.ylabel('Time len')
    plt.xlabel('Position')
    plt.show()

    n = 60 - 1 # 

    datalen = len(time[n])
    trainlen = int(len(time[n])*0.8)
    testlen = datalen - trainlen

    training_timeseries = data[n][:trainlen]
    test_timeseries_set = [data[n][trainlen:]]

    spectral_radius_values = [0.5]
    sparsity_values = [0.1]

    esn, errors = optimal_esn(
        training_timeseries, 
        test_timeseries_set, 
        spectral_radius_values=spectral_radius_values, 
        sparsity_values=sparsity_values,
        n_reservoir=1500, 
        return_errors=True, 
        random_seed_starts_at=42, 
        trial_number=1
    )

    model = EsnForecaster(
        n_reservoir=1500,
        spectral_radius=spectral_radius_values[0],
        sparsity=sparsity_values[0],
        regularization='noise',
        lambda_r=0.001,
        random_state=42
    )

    model.fit(training_timeseries)

    N = 1

    plt.figure(figsize=(25,4*N))

    for i in range(N):
        plt.subplot(N,1,i+1)
        
        plt.plot(time[n][:trainlen], training_timeseries[:, i], label='Train')
        plt.plot(time[n][trainlen:], test_timeseries_set[0][:, i], label='Test')
        plt.plot(time[n][trainlen:], esn.predict(np.ones(testlen))[:, i], label='Pred optimap_esn')
        plt.plot(time[n][trainlen:], model.predict(testlen)[:, i], label='Pred EsnForecaster')

        plt.xlabel(r'$L_{timeseries}$', fontsize=16)
        plt.ylabel(fr'$U_{i+1}$', fontsize=16)
        
        plt.legend()
        plt.grid()
    plt.show()