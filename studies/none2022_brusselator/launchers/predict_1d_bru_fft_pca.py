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
)


def a_coeff(a_hat):
    return 2 * np.real(a_hat)


def b_coeff(a_hat):
    return -2 * np.imag(a_hat)


def allab(x, coef_a, coef_b):
    result = 0
    for i in range(len(coef_a)):
        result += coef_a[i] * np.cos(i * x) + coef_b[i] * np.sin(i * x)
    return result


def get_spectrum_old(n_data):
    x_ = np.linspace(-np.pi, np.pi, n_data.shape[0], endpoint=False)
    a_hat_coeffs = (-1) ** (np.arange(len(x_))) * np.fft.fft(n_data) / len(x_)
    a_coeffs = a_coeff(a_hat_coeffs[: int(len(x_) // 2)])
    b_coeffs = b_coeff(a_hat_coeffs[: int(len(x_) // 2)])
    return np.r_[a_coeffs, b_coeffs]


def get_spectrum(n_data):
    x_ = np.linspace(-np.pi, np.pi, n_data.shape[0], endpoint=False)
    coeffs = np.fft.fft(n_data)[
        : int(len(x_) // 2) + 1
    ]  # +1 is important to catch Nyquist frequency
    a_coeffs = np.real(coeffs)
    b_coeffs = np.imag(coeffs)[1:-1]
    return (
        np.r_[a_coeffs, b_coeffs] / 200
    )  # coefficient 200 is just a scaler. Without scaling, ESN does not train well


def get_from_spectrum_old(x_sh, coeffs):
    a_coeffs = coeffs[: coeffs.shape[0] // 2]
    b_coeffs = coeffs[coeffs.shape[0] // 2 :]
    x_ = np.linspace(-np.pi, np.pi, x_sh, endpoint=False)  # coeffs.shape[0]
    return allab(x_, a_coeffs, b_coeffs) - a_coeffs[0] / 2


def get_from_spectrum(x_sh, coeffs):
    a_coeffs = coeffs[: coeffs.shape[0] // 2 + 1]
    b_coeffs = coeffs[coeffs.shape[0] // 2 + 1 :]
    complex_coeffs = a_coeffs + 1j * np.r_[[0], b_coeffs, [0]]
    complete_complex_coeffs = np.r_[
        complex_coeffs, np.conjugate(np.flip(complex_coeffs[1:-1]))
    ]
    return 200 * np.real_if_close(
        np.fft.ifft(complete_complex_coeffs), tol=100
    )  # coefficient 200 is just a scaler. Without scaling, ESN does not train well


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

    # находим спектры
    # coeffs_data = np.zeros((18001, 512))  # TODO: Anton's changes
    coeffs_data = np.zeros((18001, 512))
    for i in range(u_v_concat.shape[0]):
        n_data = u_v_concat[i]
        coeffs = get_spectrum(n_data)
        coeffs_data[i] = coeffs

    # ESN
    rand = 10
    np.random.seed(rand)
    esn_type = "stand"  #'stand' 'opt_param'
    if esn_type == "stand":
        esn_bru_coeff = EsnForecaster(
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
        esn_bru_coeff = EsnForecaster(
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

    pca_num = 8
    pca = PCA(n_components=pca_num)
    X_pca_reduced = pca.fit_transform(coeffs_data)

    # разделяем на обучающую и тестовую выборки
    time_train = 10000
    train_data_pca = np.array(X_pca_reduced[:time_train, :])
    # print(train_data_pca.shape)

    # standard fitting
    error = esn_bru_coeff.fit(train_data_pca, inspect=True)

    time_predict = coeffs_data.shape[0] - time_train
    prediction_coeffs = esn_bru_coeff.predict(time_predict, inspect=True)
    # print(prediction_coeffs.shape)

    X_pca_returned = pca.inverse_transform(prediction_coeffs)
    # print(X_pca_returned.shape)

    data_train_ret = pca.inverse_transform(train_data_pca)
    # print(data_train_ret.shape)

    # возвращаемся от спектров train
    data_train = np.zeros((time_train, 512))
    for i in range(data_train_ret.shape[0]):
        data_train[i] = get_from_spectrum(x.shape[0] * 2, data_train_ret[i])

    print(data_train.shape)

    # возвращаемся от спектров predict
    data_return = np.zeros((time_predict, 512))
    for i in range(X_pca_returned.shape[0]):
        data_return[i] = get_from_spectrum(x.shape[0] * 2, X_pca_returned[i])

    print(data_return.shape)

    plot_data_train_pred(
        u_v_concat,
        data_train,
        data_return,
        t,
        2,
        filename=f"u_v_ot_t_esn_{esn_type}_{pca_num}.png",
    )

    plot_data_phase_traj(
        u_v_concat,
        data_train,
        data_return,
        2,
        filename=f"phase_tr_2_esn_{esn_type}_{pca_num}.png",
    )  # , figsz=(4,4)

    t_pred = np.arange(
        20 + 0.01 * time_train, 20 + 0.01 * (time_train + time_predict), 0.01
    )

    plot_data_train_pred_mesh(
        x,
        t_pred,
        data_return[:, :256],
        data_return[:, 256:],
        u_v_concat,
        time_train,
        figsz=(6, 2.5),
        filename=f"pred_colormesh_esn_{esn_type}_{pca_num}.png",
    )
