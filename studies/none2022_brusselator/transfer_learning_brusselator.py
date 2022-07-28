import os
import sys
sys.path.append(os.getcwd())
sys.path.append('/Users/tony/reps/github/dynolab/skesn')
import pickle
import json

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skesn.esn import EsnForecaster, UpdateModes
from skesn.cross_validation import ValidationBasedOnRollingForecastingOrigin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import restools
from comsdk.research import Research
from comsdk.misc import load_from_json
from thequickmath.reduced_models.models import BrusselatorModel, rk4_timestepping


if __name__ == '__main__':
    plt.style.use('resources/default.mplstyle')

    delta_t = 10**(-3)
    coarse_delta_t = 10**(-2)
    t_max = 100
    b = 4.0
    brusselator = BrusselatorModel(a=1.0, b=b)
    ts = rk4_timestepping(brusselator,
                          ic=np.array([1.0, 1.0]),
                          delta_t=delta_t,
                          n_steps=int(round(t_max / delta_t)),
                          time_skip=int(round(coarse_delta_t / delta_t)))
    ts = np.log(ts[:-1, :])
    #ts = ts[:-1, :]
    times = np.linspace(0, t_max, int(round(t_max / coarse_delta_t)), endpoint=False)
    coord_names = [r'$x$', r'$y$']
    esn_dt = coarse_delta_t
    ss = StandardScaler(with_mean=False, with_std=False)
    ts = ss.fit_transform(ts)
    model = EsnForecaster(
        n_reservoir=200,
        spectral_radius=0.2,
        sparsity=0.8,
        #regularization='noise',
        #lambda_r=0.1,
        regularization='l2',
        lambda_r=0.0001,
        #lambda_r=0.01,
        in_activation='tanh',
        out_activation='identity',
        use_additive_noise_when_forecasting=False,
        random_state=None,
        use_bias=True)

    # Plot long-term prediction
    n_prediction = int(ts.shape[0])
    future_times = np.arange(times[-1] + coarse_delta_t, times[-1] + coarse_delta_t * (n_prediction + 1), coarse_delta_t)
    model.fit(ts)
    fig, axes_original = plt.subplots(ts.shape[1]*3, 1, figsize=(12, 6), sharex=True)
    axes = axes_original[:2]
    for i in range(ts.shape[1]):
        axes[i].plot(times, np.exp(ss.inverse_transform(ts)[:, i]),
        #axes[i].plot(times, ts[:, i],
                     linewidth=2,
                     label='True')
        axes[i].plot(future_times, np.exp(ss.inverse_transform(model.predict(n_prediction))[:, i]),
        #axes[i].plot(future_times, model.predict(n_prediction)[:, i],
                     linewidth=2,
                     label='Prediction')
        ylim = axes[i].get_ylim()
        axes[i].fill_between(times, y1=ylim[0], y2=ylim[1], color='tab:blue', alpha=0.2, zorder=-10)
        axes[i].set_ylabel(coord_names[i], fontsize=12)
        axes[i].grid()
        axes[i].legend()

#    t_lims = axes[0].get_xlim()
    #for ax in axes:
    #    ax.set_rasterization_zorder(0)
    #axes[-1].set_xlabel(r'$t$')
    #plt.tight_layout()
    #plt.savefig(f'esn_prediction_of_brusselator_b_{int(b)}.eps', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    #plt.show()

    # Transfer learning from b = 4 to b = 6
    
    b = 6.0
    t_max = int(round(t_max / 4.))
    brusselator = BrusselatorModel(a=1.0, b=b)
    ts = rk4_timestepping(brusselator,
                          ic=np.array([1.0, 1.0]),
                          delta_t=delta_t,
                          n_steps=int(round(t_max / delta_t)),
                          time_skip=int(round(coarse_delta_t / delta_t)))
    ts = np.log(ts[:-1, :])
    #ts = ts[:-1, :]
    times = np.linspace(0, t_max, int(round(t_max / coarse_delta_t)), endpoint=False)
    esn_dt = coarse_delta_t
    ts = ss.fit_transform(ts)
    model.update(ts, mode=UpdateModes.transfer_learning, mu=1e-8)

    # Plot long-term prediction
    
    n_prediction = int(ts.shape[0]*7)
    future_times = np.arange(times[-1] + coarse_delta_t, times[-1] + coarse_delta_t * (n_prediction + 1), coarse_delta_t)
    axes = axes_original[2:4]
    for i in range(ts.shape[1]):
        axes[i].plot(times, np.exp(ss.inverse_transform(ts)[:, i]),
        #axes[i].plot(times, ts[:, i],
                     linewidth=2,
                     label='True')
        axes[i].plot(future_times, np.exp(ss.inverse_transform(model.predict(n_prediction))[:, i]),
        #axes[i].plot(future_times, model.predict(n_prediction)[:, i],
                     linewidth=2,
                     label='Prediction')
        ylim = axes[i].get_ylim()
        axes[i].fill_between(times, y1=ylim[0], y2=ylim[1], color='tab:blue', alpha=0.2, zorder=-10)
        axes[i].set_ylabel(coord_names[i], fontsize=12)
        axes[i].grid()
        axes[i].legend()

#    t_lims = axes[0].get_xlim()
    #for ax in axes:
    #    ax.set_rasterization_zorder(0)
    #axes[-1].set_xlabel(r'$t$')
    #plt.tight_layout()
    #plt.savefig(f'esn_prediction_of_brusselator_b_{int(b)}.eps', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    #plt.show()

    # Transfer learning from b = 6 to b = 8

    b = 8.0
    t_max = int(round(t_max * 1.5))  # * 2
    brusselator = BrusselatorModel(a=1.0, b=b)
    ts = rk4_timestepping(brusselator,
                          ic=np.array([1.0, 1.0]),
                          delta_t=delta_t,
                          n_steps=int(round(t_max / delta_t)),
                          time_skip=int(round(coarse_delta_t / delta_t)))
    ts = np.log(ts[:-1, :])
    #ts = ts[:-1, :]
    times = np.linspace(0, t_max, int(round(t_max / coarse_delta_t)), endpoint=False)
    esn_dt = coarse_delta_t
    ts = ss.fit_transform(ts)
    model.update(ts, mode=UpdateModes.transfer_learning, mu=1e-7)

    # Plot long-term prediction
    
    n_prediction = int(ts.shape[0]*4)
    future_times = np.arange(times[-1] + coarse_delta_t, times[-1] + coarse_delta_t * (n_prediction + 1), coarse_delta_t)
    axes = axes_original[4:]
    for i in range(ts.shape[1]):
        axes[i].plot(times, np.exp(ss.inverse_transform(ts)[:, i]),
        #axes[i].plot(times, ts[:, i],
                     linewidth=2,
                     label='True')
        axes[i].plot(future_times, np.exp(ss.inverse_transform(model.predict(n_prediction))[:, i]),
        #axes[i].plot(future_times, model.predict(n_prediction)[:, i],
                     linewidth=2,
                     label='Prediction')
        ylim = axes[i].get_ylim()
        axes[i].fill_between(times, y1=ylim[0], y2=ylim[1], color='tab:blue', alpha=0.2, zorder=-10)
        axes[i].set_ylabel(coord_names[i], fontsize=12)
        axes[i].grid()
        axes[i].legend()

#    t_lims = axes[0].get_xlim()
    for ax in axes_original:
        ax.set_rasterization_zorder(0)
    axes[-1].set_xlabel(r'$t$')
    plt.tight_layout()
    plt.savefig(f'esn_prediction_of_brusselator_b_{int(b)}.eps', dpi=200)
    #gs.tight_layout(fig, rect=[0, 0, 1.5, 1.5])
    plt.show()
