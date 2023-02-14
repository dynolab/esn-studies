from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from skesn.esn import EsnForecaster
from comsdk.research import Research

from studies.none2022_brusselator.extensions import (
    preprocess_1d_bru_data,
    plot_data_phase_traj,
    plot_data_train_pred,
    plot_data_train_pred_mesh,
    get_spectrum_1d,
    get_from_spectrum_1d,
)


if __name__ == "__main__":
    plt.style.use("resources/default.mplstyle")

    task = 1
    res_id = "BRU"
    res = Research.open(res_id)
    task_path = res.get_task_path(task)
    data_num = 2
    filename = Path(task_path) / f"brusselator1DB_{data_num}.npz"
    data = np.load(filename)
    x, t, u_v_concat = preprocess_1d_bru_data(data)

    coeffs_data = np.zeros((18001, 512))
    abs_coeffs_data = np.zeros((18001, 257))
    for i in range(u_v_concat.shape[0]):
        n_data = u_v_concat[i]
        coeffs = get_spectrum_1d(n_data)
        abs_coeffs = get_spectrum_1d(n_data, treat_as_amplitude=True)
        coeffs_data[i] = coeffs
        abs_coeffs_data[i] = abs_coeffs

    # Fourier spectrum
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x_values = np.arange(abs_coeffs_data.shape[1])
    # ax.semilogy(x_values, np.mean(abs_coeffs_data, axis=0))
    ax.semilogy(x_values, abs_coeffs_data[10000, :])
    ax.grid()
    ax.set_xlabel("Component", fontsize=12)
    ax.set_ylabel("Explained variance percent", fontsize=12)
    plt.tight_layout()
    plt.savefig("fourier_spectrum_1d.eps")
    plt.show()

    # Explained variance ratio graph
    pca_num = 20
    pca = PCA(n_components=pca_num)
    X_pca_reduced = pca.fit_transform(coeffs_data)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x_values = np.arange(1, pca.n_components_ + 1)
    ax.semilogy(x_values, pca.explained_variance_ratio_, "o--")
    ax.set_xticks(x_values)
    ax.grid()
    ax.set_xlabel("Component", fontsize=12)
    ax.set_ylabel("Explained variance percent", fontsize=12)
    plt.tight_layout()
    plt.savefig("pca_explained_variance.eps")
    plt.show()

    # PCA/POD modes
    pca_num = 8
    pca = PCA(n_components=pca_num)
    X_pca_reduced = pca.fit_transform(coeffs_data)
    physical_components = np.zeros((pca.n_components_, 512))
    for i in range(physical_components.shape[0]):
        physical_components[i] = get_from_spectrum_1d(None, pca.components_[i])
    fig, axes = plt.subplots(4, 2, figsize=(12, 8))
    x_values = np.arange(physical_components.shape[1])
    for i, ax in enumerate(axes.reshape(-1)):
        ax.plot(x_values, physical_components[i])
        ax.set_title(f"Component #{i+1}", fontsize=12, usetex=False)
        ax.grid()
    for ax in axes[3, :]:
        ax.set_xlabel(r"$x$", fontsize=12)
    plt.tight_layout()
    plt.savefig("pca_components_fourier.eps")
    plt.show()
