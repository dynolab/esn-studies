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
)

if __name__ == "__main__":
    plt.style.use("resources/default.mplstyle")

    task = 2
    res_id = "BRU"
    res = Research.open(res_id)
    task_path = res.get_task_path(task)
    filename = Path(task_path) / "brusselator2DA_1.9B_4.8.npz"
    data = np.load(filename)

    # PCA/POD decomposition
    data_u = data["u"]
    data_v = data["v"]
    time_dim = data_u.shape[0]
    full_dim = np.prod(data_u.shape)
    full_space_dim = np.prod(data_u.shape[1:])
    pca = PCA(n_components=2)
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
            lambda_r=1e-13,
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
