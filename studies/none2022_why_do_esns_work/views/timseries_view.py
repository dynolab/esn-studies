import os
import sys
sys.path.append(os.getcwd())

import restools
from comsdk.research import Research
from skesn.esn import EsnForecaster

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

if __name__ == '__main__':
    res = Research.open('RC_MOEHLIS')

    task_num = 55
    data5 = []
    time = []
    n_files = 100

    for num_data in range(n_files):
        with open(os.path.join(res.get_task_path(task_num), str(num_data+1)), 'rb') as a:
            b = pickle.load(a)
        data5.append(b['timeseries'])
        time.append(b['time'])

    plt.figure(figsize=(20,8))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        for j in range(100):
            plt.plot(
                np.array(data5[j])[:, i], 
                '-',
                color='#0000FF11'
                );
        plt.xlabel(fr'$L_{"{timeseries}"}$')
        plt.ylabel(fr'$U_{i+1}$')
        #plt.semilogx()
plt.show()