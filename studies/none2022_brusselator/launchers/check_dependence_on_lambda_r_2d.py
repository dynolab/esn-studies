from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from skesn.esn import EsnForecaster
from comsdk.research import Research
from tqdm import tqdm

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

    # Train/test split
    ss = StandardScaler()
    u_v_concat = ss.fit_transform(u_v_concat)
    split_ratio = 0.9
    time_train = int(time_dim * split_ratio)
    train_data = np.array(u_v_concat[:time_train, :])
    test_data = np.array(u_v_concat[time_train:, :])
    test_data = ss.inverse_transform(test_data)

    # ESN
    rand_seeds = np.arange(1, 11)
    lambda_r_values = np.logspace(-16, 2, 19, base=10)
    rmse_values = np.zeros((len(lambda_r_values), len(rand_seeds)))
    for l_i, lambda_r in tqdm(enumerate(lambda_r_values), total=len(lambda_r_values)):
        for r_i, rand in enumerate(rand_seeds):
            np.random.seed(rand)
            esn_type = "stand"  #'stand' 'opt_param'
            if esn_type == "stand":
                esn_bru_uv = EsnForecaster(
                    n_reservoir=1500,
                    spectral_radius=0.95,
                    sparsity=0,
                    regularization="l2",
                    lambda_r=lambda_r,
                    in_activation="tanh",
                    out_activation="identity",
                    use_additive_noise_when_forecasting=False,
                    random_state=rand,
                    use_bias=True,
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

            # standard fitting
            error = esn_bru_uv.fit(train_data, inspect=False)
            time_predict = u_v_concat.shape[0] - time_train
            prediction_coeffs = esn_bru_uv.predict(time_predict, inspect=False)

            # u_v_concat = ss.inverse_transform(u_v_concat)
            # train_data = ss.inverse_transform(train_data)

            prediction_coeffs = ss.inverse_transform(prediction_coeffs)
            rmse = np.sqrt(mean_squared_error(test_data, prediction_coeffs))
            rmse_values[l_i, r_i] = rmse
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.loglog(lambda_r_values, np.mean(rmse_values, axis=1), "o--", linewidth=2)
    ax.set_xlabel(r"$\lambda_r$")
    ax.set_ylabel("RMSE")
    ax.grid()
    plt.tight_layout()
    plt.savefig("dependence_on_lambda_r_2d.eps")
    plt.show()
