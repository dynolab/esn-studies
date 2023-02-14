from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from skesn.esn import EsnForecaster

from comsdk.research import Research
from studies.none2022_brusselator.extensions import (
    get_spectrum_2d,
    get_wavelet_spectrum_2d,
)

if __name__ == "__main__":
    plt.style.use("resources/default.mplstyle")

    task = 2
    res_id = "BRU"
    res = Research.open(res_id)
    task_path = res.get_task_path(task)
    filename = Path(task_path) / "brusselator2DA_1.9B_4.8.npz"
    data = np.load(filename)

    # Fourier spectrum
    u_coeffs = get_spectrum_2d(data["u"][300, :, :], treat_as_amplitude=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    im = ax.imshow(np.log10(u_coeffs))
    plt.colorbar(im)
    plt.show()

    # Wavelet spectrum
    data_ = data["u"][300, :, :]
    for l in range(5):
        u_coeffs = get_wavelet_spectrum_2d(data_)
        data_ = u_coeffs[0]
        fig, axes = plt.subplots(1, 4, figsize=(12, 6))
        for i, ax in enumerate(axes):
            if i == 0:
                im = ax.imshow(u_coeffs[0])
            else:
                im = ax.imshow(u_coeffs[1][i - 1])
        # plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(f"wavelet_level_{l+1}.png", dpi=100)
        plt.show()

    # PCA/POD analysis
    data_u = data["u"]
    time_dim = data_u.shape[0]
    full_dim = np.prod(data_u.shape)
    full_space_dim = np.prod(data_u.shape[1:])
    data_matrix = data_u.reshape(time_dim, full_space_dim)
    pca = PCA(n_components=50)
    X_pca_reduced = pca.fit_transform(data_matrix)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x_values = np.arange(1, pca.n_components_ + 1)
    ax.semilogy(x_values, pca.explained_variance_ratio_, "o--")
    ax.set_xticks(x_values)
    ax.grid()
    ax.set_xlabel("Component", fontsize=12)
    ax.set_ylabel("Explained variance percent", fontsize=12)
    plt.tight_layout()
    plt.savefig("pca_explained_variance_2d.eps")
    plt.show()

    # PCA/POD modes
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    for i, ax in enumerate(axes.reshape(-1)):
        physical_component = pca.components_[i].reshape(data_u.shape[1:])
        im = ax.imshow(physical_component)
        ax.set_title(f"Component #{i+1}", fontsize=12, usetex=False)
    plt.tight_layout()
    plt.savefig("pca_components_2d.png", dpi=100)
    plt.show()
