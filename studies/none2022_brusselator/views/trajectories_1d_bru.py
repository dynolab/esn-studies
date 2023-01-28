from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from skesn.esn import EsnForecaster

from comsdk.research import Research


def plot_data(data, data_num, save=False):
    x = data["x"]
    u = data["u"]
    v = data["v"]
    t = data["t"]

    plt.pcolormesh(x, t, u, shading="nearest", cmap="plasma")
    plt.colorbar()
    plt.xlabel(r"$x$", fontsize=16)
    plt.ylabel(r"$t$", fontsize=16)
    plt.tight_layout()
    plt.ylim([175, 200])
    if save == True:
        plt.savefig(f"u_colormesh_{data_num}.png")
    plt.show()

    plt.plot(u[:, 2], v[:, 2])
    plt.xlabel(r"$u$", fontsize=16)
    plt.ylabel(r"$v$", fontsize=16)
    if save == True:
        plt.savefig(f"phase_tr_{data_num}.png")
    plt.show()


if __name__ == "__main__":
    plt.style.use("resources/default.mplstyle")

    task = 1
    res_id = "BRU"
    res = Research.open(res_id)
    task_path = res.get_task_path(task)
    for data_num in range(1, 6 + 1):
        filename = Path(task_path) / f"brusselator1DB_{data_num}.npz"
        data = np.load(filename)
        plot_data(data, data_num, save=True)
