import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from thequickmath.field import Space, map_to_2d_mesh
import pywt

Array1D = np.ndarray
Array2D = np.ndarray


def preprocess_1d_bru_data(data):
    data.files
    x = data["x"]
    t = data["t"]
    u = data["u"]
    v = data["v"]
    # plot_data('2')
    x_space = Space((t, x))
    x_384 = np.linspace(x[0], x[-1], v.shape[1])
    v_space = Space((t, x_384))
    v_new = map_to_2d_mesh(v, v_space, x_space)

    # concatenating u and v
    u_v_concat = np.zeros((u.shape[0], u.shape[1] + v_new.shape[1]))
    u_v_concat[:, :256] = u
    u_v_concat[:, 256:] = v_new

    return x, t, u_v_concat


def plot_data_train_pred(n_data, data_train, data_predicted, t, i, filename=None):
    fig, axes = plt.subplots(2, 1, figsize=(15, 5))
    x_ = t  # np.linspace(-np.pi, np.pi, n_data.shape[0], endpoint=False)
    for ax in axes:
        ax.plot(x_, n_data[:, i], "-", linewidth=1.5, label="True")
        ax.plot(
            x_[: data_train.shape[0]],
            data_train[:, i],
            "m--",
            linewidth=1.5,
            label="Train",
        )
        ax.plot(
            x_[data_train.shape[0] :],
            data_predicted[:, i],
            "--",
            linewidth=1.5,
            label="Prediction",
        )
        # ax.plot(x_, np.fft.ifft(a_hat_coeffs), '.-',linewidth=1)
        ax.grid()
        ax.legend()
        i += n_data.shape[1] // 2
    axes[0].set_ylabel(r"$u$", fontsize=14)
    axes[1].set_ylabel(r"$v$", fontsize=14)
    plt.xlabel(r"$t$", fontsize=14)
    if filename is not None:
        plt.savefig(filename)


#   plt.show()


def plot_data_phase_traj(
    n_data, data_train, data_predicted, i, figsz=(7, 7), filename=None
):
    fig, ax = plt.subplots(figsize=figsz)
    u = n_data[:, :256]
    v = n_data[:, 256:]
    plt.plot(u[:, i], v[:, i], "-", label="True")
    plt.plot(data_train[:, i], data_train[:, 256 + i], "m--", label="Train")
    plt.plot(data_predicted[:, i], data_predicted[:, 256 + i], "-", label="Prediction")
    plt.xlabel(r"$u$", fontsize=14)
    plt.ylabel(r"$v$", fontsize=14)
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    # plt.show()


def plot_data_train_pred_mesh(
    x, t_pred, u_pred, v_pred, u_v_concat, time_train, figsz=(10, 4), filename=None
):
    # предсказанное colormesh u
    u = u_v_concat[:, :256]
    v = u_v_concat[:, 256:]
    for field, field_pred, name in zip((u, v), (u_pred, v_pred), ("u", "v")):
        fig, ax = plt.subplots(1, 2, figsize=figsz)
        p1 = ax[0].pcolormesh(
            x,
            t_pred[: field_pred.shape[0] - 1],
            field[time_train:-1, :],
            shading="nearest",
            cmap="plasma",
        )
        p2 = ax[1].pcolormesh(
            x,
            t_pred[: field_pred.shape[0] - 1],
            field_pred[:-1, :],
            shading="nearest",
            cmap="plasma",
        )
        ax[0].set_xlabel(r"$x$")  # , fontsize=14)
        ax[1].set_xlabel(r"$x$")  # , fontsize=14)
        ax[0].set_ylabel(r"$t$")  # , fontsize=14)
        fig.colorbar(p1, ax=ax[0])
        fig.colorbar(p2, ax=ax[1])
        plt.suptitle(f"{name} and {name}_pred")
        plt.tight_layout()
        # plt.ylim([175,200])
        if filename is not None:
            plt.savefig(f"{name}_{filename}")
        # plt.show()


def get_spectrum_1d(data: Array1D, treat_as_amplitude=False) -> Array1D:
    coeffs = np.fft.fft(data)[
        : int(data.shape[0] // 2) + 1
    ]  # +1 is important to catch Nyquist frequency
    if treat_as_amplitude:
        return np.abs(coeffs) / 200
    else:
        a_coeffs = np.real(coeffs)
        b_coeffs = np.imag(coeffs)[1:-1]
        return (
            np.r_[a_coeffs, b_coeffs] / 200
        )  # coefficient 200 is just a scaler. Without scaling, ESN does not train well


def get_spectrum_2d(data: Array2D, treat_as_amplitude=False) -> Array2D:
    coeffs = np.fft.fft2(data)[
        : int(data.shape[0] // 2) + 1, : int(data.shape[1] // 2) + 1
    ]  # +1 is important to catch Nyquist frequency
    if treat_as_amplitude:
        return np.abs(coeffs) / 200
    else:
        a_coeffs = np.real(coeffs)
        b_coeffs = np.imag(coeffs)[1:-1, 1:-1]
        return (
            np.r_[a_coeffs.reshape(-1), b_coeffs.reshape(-1)] / 200
        )  # coefficient 200 is just a scaler. Without scaling, ESN does not train well


def get_wavelet_spectrum_2d(data: Array2D) -> Array2D:
    coeffs = pywt.dwt2(data, "db8", mode="periodic")
    # coeffs = pywt.dwt2(data, "haar", mode="periodic")
    return coeffs


def get_from_spectrum_1d(x_sh, coeffs):
    b_coeffs = coeffs[coeffs.shape[0] // 2 + 1 :]
    a_coeffs = coeffs[: coeffs.shape[0] // 2 + 1]
    complex_coeffs = a_coeffs + 1j * np.r_[[0], b_coeffs, [0]]
    complete_complex_coeffs = np.r_[
        complex_coeffs, np.conjugate(np.flip(complex_coeffs[1:-1]))
    ]
    return 200 * np.real_if_close(
        np.fft.ifft(complete_complex_coeffs), tol=100
    )  # coefficient 200 is just a scaler. Without scaling, ESN does not train well

def make_a_movie(field, filename = None):
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.axis('off')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    ims = []
    for i in range(field.shape[0]):
        im = ax.imshow(field[i,:,:],cmap='winter',aspect='auto')
        ims.append([im])

    if filename is not None:
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        writervideo = animation.FFMpegWriter(fps=60)
        ani.save(f"{filename}.mp4", writer=writervideo)