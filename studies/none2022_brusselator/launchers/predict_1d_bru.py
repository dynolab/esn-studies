from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
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

    task = 1
    res_id = "BRU"
    res = Research.open(res_id)
    task_path = res.get_task_path(task)
    data_num = 2
    filename = Path(task_path) / f"brusselator1DB_{data_num}.npz"
    data = np.load(filename)
    x, t, u_v_concat = preprocess_1d_bru_data(data)
    t_miss = 7  # dt = t_miss * 0.01 (both for training and prediction)
    u_v_concat = u_v_concat[::t_miss, :]
    print(" t_miss ", t_miss, "\n step ", t_miss * 0.01)

    # ESN
    rand = 10
    np.random.seed(rand)
    esn_type = "opt_param"  #'stand' 'opt_param'
    if esn_type == "stand":
        esn_bru_uv = EsnForecaster(
            n_reservoir=1500,
            spectral_radius=0.95,
            sparsity=0,
            regularization="noise",
            lambda_r=0.001,
            in_activation="tanh",
            out_activation="identity",
            use_additive_noise_when_forecasting=True,
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

    # разделяем на обучающую и тестовую выборки
    time_train = 10000 // t_miss
    train_data = np.array(u_v_concat[:time_train, :])
    # print(train_data_pca.shape)

    # standard fitting
    error = esn_bru_uv.fit(train_data, inspect=True)
    time_predict = u_v_concat.shape[0] - time_train
    prediction_coeffs = esn_bru_uv.predict(time_predict, inspect=True)

    plot_data_train_pred(
        u_v_concat,
        train_data,
        prediction_coeffs,
        t[::t_miss],
        2,
        filename=f"u_v_ot_t_esn_{esn_type}_miss_{t_miss}.png",
    )

    plot_data_phase_traj(
        u_v_concat,
        train_data,
        prediction_coeffs,
        2,
        filename=f"phase_tr_2_esn_{esn_type}_miss_{t_miss}.png",
    )  # , figsz=(4,4)

    t_step = t_miss * 0.01
    t_pred = np.arange(
        20 + t_step * time_train, 20 + t_step * (time_train + time_predict), t_step
    )

    plot_data_train_pred_mesh(
        x,
        t_pred,
        prediction_coeffs[:, :256],
        prediction_coeffs[:, 256:],
        u_v_concat,
        time_train,
        figsz=(6, 2.5),
        filename=f"pred_colormesh_esn_{esn_type}_miss_{t_miss}.png",
    )
