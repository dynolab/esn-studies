from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from skesn.esn import EsnForecaster
from skesn.cross_validation import ValidationBasedOnRollingForecastingOrigin
from skesn.weight_generators import optimal_weights_generator
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
    esn_dt = 0.01

    # ESN
    rand = 10
    np.random.seed(rand)
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

    # Plot hyperparameter grid search results based on rolling forecasting origin
    test_time_length = 2
    n_splits = 10
    v = ValidationBasedOnRollingForecastingOrigin(
        n_training_timesteps=None,
        n_test_timesteps=int(test_time_length / esn_dt),
        n_splits=n_splits,
        metric=mean_squared_error,
        # TODO: failed, no reason to use
        # initialization_strategy=optimal_weights_generator(
        #    verbose=2,
        #    range_generator=np.linspace,
        #    steps=100,
        #    hidden_std=0.5,
        #    find_optimal_input=True,
        #    thinning_step=10,
        # ),
    )
    summary, best_model = v.grid_search(
        esn_bru_uv,
        param_grid=dict(
            spectral_radius=[0.5, 0.95],
            sparsity=[0, 0.8],
            lambda_r=[0.01, 0.001],  # [0.01, 0.005, 0.001, 0.0001]
        ),
        y=u_v_concat,
        X=None,
    )
    summary_df = pd.DataFrame(summary).sort_values("rank_test_score")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharex=True)
    table_rows = []
    param_names = list(summary_df.iloc[0]["params"].keys())
    ranks = []
    test_scores = []
    for i in range(len(summary_df)):
        ranks.append(int(summary_df.iloc[i]["rank_test_score"]))
        table_rows.append(list(summary_df.iloc[i]["params"].values()))
        test_scores.append(
            np.abs(
                np.array(
                    [
                        float(summary_df.iloc[i][f"split{j}_test_score"])
                        for j in range(n_splits)
                    ]
                )
            )
        )
    ax.boxplot(test_scores)
    ax.set_yscale("log")
    ax.set_xticks([])
    ax.set_ylabel("MSE", fontsize=12)
    ax.grid()
    table_rows = [*zip(*table_rows)]
    the_table = ax.table(
        cellText=table_rows, rowLabels=param_names, colLabels=ranks, loc="bottom"
    )
    plt.tight_layout()
    #!change this path to yours!
    plt.savefig("hyperparameter_search.png")
    plt.show()
