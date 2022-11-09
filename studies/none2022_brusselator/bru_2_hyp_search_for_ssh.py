from math import isclose
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

from thequickmath.field import Space, map_to_2d_mesh
from skesn.esn import EsnForecaster
from skesn.cross_validation import ValidationBasedOnRollingForecastingOrigin
from skesn.weight_generators import optimal_weights_generator


def _bru(data_number):
    #!change this path to yours!
    filename = '/home/amamelkina/Brusselator/1-Dataset_from_Calum_1D_Brusselator/brusselator1DB_%s.npz'  %data_number
    data = np.load(filename)
    data.files
    x = data['x']
    t = data['t']
    u = data['u']
    v = data['v']
    #plot_data('2')
    x_space = Space((t,x))
    x_384 = np.linspace(x[0], x[-1], v.shape[1])
    v_space = Space((t,x_384))
    v_new= map_to_2d_mesh(v, v_space, x_space)

    #concatenating u and v
    u_v_concat = np.zeros((u.shape[0], u.shape[1]+v_new.shape[1]))
    u_v_concat[:,:256] = u
    u_v_concat[:,256:] = v_new
    
    return t, u_v_concat


if __name__ == '__main__':
    esn_dt = 0.01
    t, u_v_concat = _bru('2')
    #ESN
    rand=10
    np.random.seed(rand)
    esn_bru_uv = EsnForecaster(n_reservoir=1500,
                               spectral_radius=0.95,
                               sparsity=0,
                               regularization='noise',
                               lambda_r=0.001,
                               in_activation='tanh',
                               out_activation='identity',
                               use_additive_noise_when_forecasting=True,
                               random_state=rand,
                               use_bias=True)

    
    """
    # Plot long-term prediction
    n_prediction = int(u_v_concat.shape[0] / 3)
    dt = t[1] - t[0]
    future_times = np.arange(t[-1] + dt, t[-1] + dt * (n_prediction ), dt)
    esn_bru_uv.fit(u_v_concat)
    num_axes = 4
    fig, axes = plt.subplots(num_axes, 1, figsize=(12, 6))
    for i in range(num_axes): #u_v_concat.shape[1]
        axes[i].plot(t, u_v_concat[:,i], #ss.inverse_transform(ts)[:, i],
                     linewidth=2,
                     label='True')
        axes[i].plot(future_times, esn_bru_uv.predict(n_prediction)[:, i], #ss.inverse_transform(model.predict(n_prediction))[:, i],
                     linewidth=2,
                     label='Prediction')
        ylim = axes[i].get_ylim()
        axes[i].fill_between(t, y1=ylim[0], y2=ylim[1], color='tab:blue', alpha=0.2)
    #    axes[i].set_ylabel(coord_names[i], fontsize=12) #each figure -- 1 of 512 
        axes[i].grid()
        axes[i].legend()
    axes[0].set_title('ESN long-term prediction of the system', fontsize=16)
    axes[-1].set_xlabel(r'$t$', fontsize=12)
    plt.tight_layout()
    plt.show()

    
    # Plot short-term forecasting skill assessment based on rolling forecasting origin
    test_time_length = 5
    v = ValidationBasedOnRollingForecastingOrigin(n_training_timesteps=None,
                                                  n_test_timesteps=int(test_time_length / esn_dt),
                                                  n_splits=18,
                                                  metric=mean_squared_error)
    fig, (ax, ax_metric) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    initial_test_times = []
    metric_values = []
    ax.plot(t, u_v_concat[:, 0], linewidth=2) #ss.inverse_transform(ts)[:, 0]
    for test_index, y_pred, y_true in v.prediction_generator(esn_bru_uv,
                                                             y=u_v_concat,
                                                             X=None):
      #  y_pred = ss.inverse_transform(y_pred)
      #  y_true = ss.inverse_transform(y_true)
        ax.plot(t[test_index], y_pred[:, 0], color='tab:orange', linewidth=2)
        ax.plot([t[test_index][0]], [y_pred[0, 0]], 'o', color='tab:red')
        initial_test_times.append(t[test_index][0])
        metric_values.append(v.metric(y_true[:, 0], y_pred[:, 0]))
    ax.set_ylabel(r'$X_t$', fontsize=12)
    ax_metric.semilogy(initial_test_times, metric_values, 'o--')
    ax_metric.set_xlabel(r'$t$', fontsize=12)
    ax_metric.set_ylabel(r'MSE', fontsize=12)
    ax.grid()
    ax_metric.grid()
    plt.tight_layout()
    plt.show()"""

    # Plot hyperparameter grid search results based on rolling forecasting origin
    test_time_length = 2
    n_splits = 10
    v = ValidationBasedOnRollingForecastingOrigin(n_training_timesteps=None,
                                                  n_test_timesteps=int(test_time_length / esn_dt),
                                                  n_splits=n_splits,
                                                  metric=mean_squared_error,
                                                  initialization_strategy = optimal_weights_generator(
                                                                            verbose = 2,
                                                                            range_generator=np.linspace,
                                                                            steps = 100,
                                                                            hidden_std = 0.5,
                                                                            find_optimal_input = True,
                                                                            thinning_step = 10)
                                                  )
    summary, best_model = v.grid_search(esn_bru_uv,
                                        param_grid=dict(
                                            spectral_radius=[0.5, 0.95],
                                            sparsity=[0, 0.8],
                                            lambda_r=[0.01, 0.001] #[0.01, 0.005, 0.001, 0.0001]
                                            ),
                                        y=u_v_concat,
                                        X=None)
    summary_df = pd.DataFrame(summary).sort_values('rank_test_score')
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharex=True)
    table_rows = []
    param_names = list(summary_df.iloc[0]['params'].keys())
    ranks = []
    test_scores = []
    for i in range(len(summary_df)):
        ranks.append(int(summary_df.iloc[i]['rank_test_score']))
        table_rows.append(list(summary_df.iloc[i]['params'].values()))
        test_scores.append(np.abs(np.array([float(summary_df.iloc[i][f'split{j}_test_score']) for j in range(n_splits)])))
    ax.boxplot(test_scores)
    ax.set_yscale('log')
    ax.set_xticks([])
    ax.set_ylabel('MSE', fontsize=12)
    ax.grid()
    table_rows = [*zip(*table_rows)]
    the_table = ax.table(cellText=table_rows,
                         rowLabels=param_names,
                         colLabels=ranks,
                         loc='bottom')
    plt.tight_layout()
    #!change this path to yours!
    plt.savefig("/home/amamelkina/Brusselator/Figures/hyperparameter_search.png")
    plt.show()
