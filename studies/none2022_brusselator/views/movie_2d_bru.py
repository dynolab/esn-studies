from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from comsdk.research import Research

import sys
sys.path.insert(1, 'C:\\Users\\njuro\\Documents\\esn-studies')

from studies.none2022_brusselator.extensions import (
    make_a_movie,
)

if __name__ == "__main__":
    plt.style.use("resources/default.mplstyle")

    task = 2
    res_id = "BRU"
    res = Research.open(res_id)
    task_path = res.get_task_path(task)
    filename = Path(task_path) / 'brusselator2DA_1.9B_4.8.npz'
    data = np.load(filename)

    x = data['x']
    t = data['t']
    u = data['u']
    v = data['v']
    
    make_a_movie(u, filename = 'movie_bru')
    print("Done!")
