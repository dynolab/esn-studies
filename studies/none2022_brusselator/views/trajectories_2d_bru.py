from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from skesn.esn import EsnForecaster

from comsdk.research import Research


def plot_data(data, save=True):
    x = data["x"]
    u = data["u"]
    v = data["v"]
    t = data["t"]

    X_, Y_ = np.meshgrid(x, x, indexing="ij")
    plt.pcolormesh(X_, Y_, u[100, :, :], shading="nearest", cmap="plasma")
    plt.colorbar()
    plt.xlabel(r"$x$", fontsize=16)
    plt.ylabel(r"$y$", fontsize=16)
    plt.tight_layout()
    # plt.ylim([175, 200])
    if save == True:
        plt.savefig("u_colormesh_2d.png")
    plt.show()

    plt.plot(u[:, 2, 2], v[:, 2, 2])
    plt.xlabel(r"$u$", fontsize=16)
    plt.ylabel(r"$v$", fontsize=16)
    if save == True:
        plt.savefig("phase_tr_2d.png")
    plt.show()


if __name__ == "__main__":
    plt.style.use("resources/default.mplstyle")

    task = 2
    res_id = "BRU"
    res = Research.open(res_id)
    task_path = res.get_task_path(task)
    filename = Path(task_path) / "brusselator2DA_1.9B_4.8.npz"
    data = np.load(filename)
    plot_data(data, save=True)
