from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skesn.esn import EsnForecaster
from comsdk.research import Research

from studies.none2022_brusselator.extensions import (
    preprocess_1d_bru_data,
    plot_data_phase_traj,
    plot_data_train_pred,
    plot_data_train_pred_mesh,
    make_a_movie,
)


PLOT_TRAJECTORIES = False
PLOT_2D_SNAPSHOTS = True
MAKE_A_MOVIE = False


if __name__ == "__main__":
    plt.style.use("resources/default.mplstyle")

    task = 2
    res_id = "BRU"
    res = Research.open(res_id)
    task_path = res.get_task_path(task)
    filename = Path(task_path) / "brusselator2DA_1.9B_4.8.npz"
    data = np.load(filename)
    n_pod_components = 2

    # PCA/POD decomposition
    data_u = data["u"]
    data_v = data["v"]
    time_dim = data_u.shape[0]
    full_dim = np.prod(data_u.shape)
    full_space_dim = np.prod(data_u.shape[1:])
    pca = PCA(n_components=n_pod_components)
    X_u = data_u.reshape(time_dim, full_space_dim)
    X_v = data_v.reshape(time_dim, full_space_dim)
    X_u_pca_reduced = pca.fit_transform(X_u)
    X_v_pca_reduced = pca.fit_transform(X_v)
    u_v_concat = np.concatenate((X_u_pca_reduced, X_v_pca_reduced), axis=1)

    # ESN
    rand = 10
    np.random.seed(rand)
    esn_type = "stand"  #'stand' 'opt_param'
    if esn_type == "stand":
        esn_bru_uv = EsnForecaster(
            n_reservoir=1500,
            spectral_radius=0.95,
            sparsity=0,
            regularization="l2",
            lambda_r=1e-5,
            in_activation="tanh",
            out_activation="identity",
            use_additive_noise_when_forecasting=False,
            random_state=rand,
            use_bias=True,
            #            n_reservoir=1500,
            #            spectral_radius=0.95,
            #            sparsity=0,
            #            regularization="noise",
            #            lambda_r=0.001,
            #            in_activation="tanh",
            #            out_activation="identity",
            #            use_additive_noise_when_forecasting=True,
            #            random_state=rand,
            #            use_bias=True,
        )
    elif esn_type == "opt_param":
        esn_bru_uv = EsnForecaster(
            n_reservoir=1500,
            spectral_radius=0.95,
            sparsity=0.2,
            regularization="noise",
            lambda_r=0.005,
            in_activation="tanh",
            out_activation="identity",
            use_additive_noise_when_forecasting=True,
            random_state=rand,
            use_bias=True,
        )

    # Train/test split
    ss = StandardScaler()
    u_v_concat = ss.fit_transform(u_v_concat)
    split_ratio = 0.9
    time_train = int(time_dim * split_ratio)
    train_data = np.array(u_v_concat[:time_train, :])
    # print(train_data_pca.shape)

    # standard fitting
    error = esn_bru_uv.fit(train_data, inspect=True)

    if PLOT_TRAJECTORIES:
        time_predict = u_v_concat.shape[0] - time_train
        prediction_coeffs = esn_bru_uv.predict(time_predict, inspect=True)
        u_v_concat = ss.inverse_transform(u_v_concat)
        train_data = ss.inverse_transform(train_data)
        prediction_coeffs = ss.inverse_transform(prediction_coeffs)
        t = np.arange(u_v_concat.shape[0])
        plot_data_train_pred(
            u_v_concat,
            train_data,
            prediction_coeffs,
            t,
            0,
            filename=f"u_v_ot_t_esn_{esn_type}_2d.png",
        )

    if PLOT_2D_SNAPSHOTS:
        time_predict = u_v_concat.shape[0] - time_train
        t_to_plot_i = time_predict - 1
        prediction_coeffs = esn_bru_uv.predict(time_predict, inspect=True)
        prediction_coeffs = ss.inverse_transform(prediction_coeffs)
        pca.fit(X_u)
        u_pred = pca.inverse_transform(
            prediction_coeffs[:, : int(prediction_coeffs.shape[1] // 2)]
        )
        u_pred = u_pred.reshape((time_predict, *(data_u.shape[1:])))
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # im = ax.imshow(u_pred[t_to_plot_i, :, :])
        im = ax.imshow(data_u[-1, :, :])
        # ax.set_title(f"{n_pod_components} POD models", fontsize=12, usetex=False)
        ax.set_title(f"Original", fontsize=12, usetex=False)
        plt.tight_layout()
        # plt.savefig(f"u_im_esn_n_pod_{n_pod_components}.png", dpi=100)
        plt.savefig(f"u_im_orig.png", dpi=100)

    if MAKE_A_MOVIE:
        time_predict = u_v_concat.shape[0]
        prediction_coeffs = esn_bru_uv.predict(time_predict, inspect=True)
        prediction_coeffs = ss.inverse_transform(prediction_coeffs)
        pca.fit(X_u)
        u_pred = pca.inverse_transform(
            prediction_coeffs[:, : int(prediction_coeffs.shape[1] // 2)]
        )
        u_pred = u_pred.reshape((time_predict, *(data_u.shape[1:])))
        make_a_movie(u_pred, filename=f"movie_esn_2d_bru_n_pod_{n_pod_components}")
