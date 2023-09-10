from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from comsdk.research import Research

from studies.none2022_brusselator.extensions import (
    make_a_movie,
)

if __name__ == "__main__":
    plt.style.use("resources/default.mplstyle")

    # task = 3
    # task = 4
    # task = 5
    # task = 6
    # task = 7 
    # task = 8 
    # task = 9 
    # task = 10 
    task = 11
    res_id = "BRU"
    res = Research.open(res_id)
    task_path = res.get_task_path(task)

    # filename = Path(task_path) / "brusselator2DA1.0_B2.1.npz" # 3
    # filename = Path(task_path) / "brusselator2DA1.0_B2.5.npz" # 4
    # filename = Path(task_path) / "brusselator2DA1.0_B3.0.npz" # 5 
    # filename = Path(task_path) / "brusselator2DA1.0_B3.5.npz" # 6
    # filename = Path(task_path) / "brusselator2DA1.0_B4.0.npz" # 7
    # filename = Path(task_path) / "brusselator2DA1.0_B2.5_seed1.npz" # 8
    # filename = Path(task_path) / "brusselator2DA1.0_B2.5_seed2.npz" # 9
    # filename = Path(task_path) / "brusselator2DA1.0_B2.5_seed3.npz" # 10
    filename = Path(task_path) / "brusselator2DA1.0_B2.5_seed4.npz" # 11

    # task = 2
    # res_id = "BRU"
    # res = Research.open(res_id)
    # task_path = res.get_task_path(task)
    # filename = Path(task_path) / 'brusselator2DA_1.9B_4.8.npz'
    data = np.load(filename)

    # x = data['x']
    # t = data['t']
    u = data['u']
    # v = data['v']
    
    make_a_movie(u, filename = f'movie_bru_task_{task}')
    print("Done!")
    import winsound
    freq = 500 # Set frequency To 2500 Hertz
    dur = 100 # Set duration To 1000 ms == 1 second
    winsound.Beep(freq, dur)
