import os
import sys
sys.path.append(os.getcwd())


import matplotlib.pyplot as plt
import pickle
import numpy as np

if __name__ == '__main__':
    task_num = 51
    data5 = []
    time = []
    n_files = 100

    for num_data in range(n_files):
        with open(os.path.join(res.get_task_path(task_num), str(num_data+1)), 'rb') as a:
            b = pickle.load(a)
        data5.append(b['timeseries'])
        time.append(b['time'])


    plt.figure(figsize=(10,10))
    ui = 0
    uj = 1
    for j in range(n_files):
        plt.plot(np.array(data5[j])[:, ui], np.array(data5[j])[:, uj], '-');

    plt.xlabel(fr'$U_{ui+1}$')
    plt.ylabel(fr'$U_{uj+1}$')

    plt.show()