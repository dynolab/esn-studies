import numpy as np
from skesn.esn import EsnForecaster, UpdateModes

import argparse
import os
from tqdm import tqdm
from math import isclose
from scipy.stats import wasserstein_distance

from skesn.weight_generators import optimal_weights_generator
from skesn.esn_controllers import *
from skesn.data_preprocess import ToNormalConverter

from comsdk.research import Research

def forecasting(model, data, controls = None):
    samples = data.shape[0]
    T = samples//2
    steps = (T//10-10)
    
    error = 0
    if(controls is not None): controls = controls[:10]
    for j in range(steps):
        model.update(data[T+j*10:T+10+j*10], controls, mode=update_modes.synchronization)
        output = model.predict(10, controls, inspect=False)
        error += np.mean((output - data[T+10+j*10:T+20+j*10])**2)
    error = error / steps

    return error

def _chui_moffatt(x_0, dt, t_final, xi = 1.):
    alpha_ = 1.5
    omega_ = 1.
    eta_ = 4.
    kappa_ = 4.
    def rhs(x):
        f_ = np.zeros(5)
        f_[0] = alpha_ * (-eta_ * x[0] + omega_ * x[1] * x[2])
        f_[1] = -eta_ * x[1] + omega_ * x[0] * x[2]
        f_[2] = kappa_ * (x[3] - x[2] - x[0] * x[1])
        f_[3] = -x[3] + xi * x[2] - x[4] * x[2]
        f_[4] = -x[4] + x[3] * x[2]
        return f_

    times = np.arange(0, t_final, dt)
    ts = np.zeros((len(times), 5))
    ts[0, :] = x_0
    cur_x = x_0
    dt_integr = 10**(-3)
    n_timesteps = int(np.ceil(dt / dt_integr))
    dt_integr = dt / n_timesteps
    for i in range(1, n_timesteps*len(times)):
        cur_x = cur_x + dt_integr * rhs(cur_x)
        saved_time_i = i*dt_integr / dt
        if isclose(saved_time_i, np.round(saved_time_i)):
            saved_time_i = int(np.round(i*dt_integr / dt))
            ts[saved_time_i, :] = cur_x
    return ts, times

if __name__ == "__main__":
    task = 0
    res_id = "CTL"
    res = Research.open(res_id)
    task_path = res.get_task_path(task)

    parser = argparse.ArgumentParser(description="Experiments with chui model")
    parser.add_argument("--type", choices=["train_orig", "train_proc", "train_log", "all"], default="all", help="Type of the experiment")
    parser.add_argument("--controller", choices=["inject", "transfer", "homotopy_simple", "homotopy_transfer", "all"], default="all", help="Type of controller")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--xi", type=int, default=32, help="Value of the analyzed xi")
    parser.add_argument("--regenerate_data", action='store_true', help="Regenerate training data")

    args = parser.parse_args()

    if(args.regenerate_data or not os.path.exists(os.path.join(task_path, "data_chui.npy"))):
        np.random.seed(args.seed)
        print("Data generation")
        data = np.zeros((20, 20, 5000, 5))
        pbar = tqdm(total=400, position=0)
        for j in range(20):
            for rho in range(20):
                y0 = np.random.rand(5, )
                ts, time = _chui_moffatt(y0, 5e-4, 220.5, rho*2+2)
                data[j, rho] = ts[1000::80][500:]
                pbar.update(1)
        time = time[:-1000:80]
        np.save(os.path.join(task_path, "data_chui.npy"), data)

    time = np.arange(0, 200, 0.04)
    data_orig = np.load(os.path.join(task_path, "data_chui.npy"))
    data = data_orig.copy()

    if(args.type == "all"):
        tr_types = ["train_orig", "train_proc", "train_log"]
    else: tr_types = [args.type]
    if(args.controller == "all"):
        tr_controllers = ["inject", "transfer", "homotopy_simple", "homotopy_transfer"]
    else: tr_controllers = [args.controller]

    for tr_type in tr_types:
        for tr_controller in tr_controllers:
            print("Train for (%s, %s)..." % (tr_type, tr_controller))

            MED, STD = np.mean(data[0, 10, 500:], axis=0), np.std(data[0, -1, 500:], axis=0)

            for i in range(len(data)):
                for j in range(len(data[i])):
                    for k in range(5):
                        if(not (k == 1 and ("_proc" in tr_type or "_log" in tr_type))): 
                            data[i, j, :, k] = (data[i, j, :, k] - MED[k]) / STD[k]

            if("_proc" in tr_type): 
                scaler = ToNormalConverter().fit(data[:4, 15:, :, 1].reshape(-1))
                for i in range(len(data)):
                    for j in range(len(data[i])):
                        data[i, j, :, 1] = scaler.transform(data[i, j, :, 1])

            if("_log" in tr_type): 
                for i in range(len(data)):
                    for j in range(len(data[i])):
                        data[i, j, :, 1] = np.log(data[i, j, :, 1])
                MED1, STD1 = np.mean(data[0, 15:, 500:, 1]), np.std(data[0, 15:, 500:, 1])
                data[:, :, :, 1] = (data[:, :, :, 1] - MED1) / STD1

            controls = (range(0,20) * np.ones((1,5000,1))).T/9.5 - 1
            if(tr_controller == "transfer"): ANC = [14]
            else: ANC = [5,10,14,18]

            np.random.seed(args.seed)
            N = 20
            w_errors = np.zeros((N, 20, ))
            f_errors = np.zeros((N, 20, ))
            print("Network training...")
            pbar = tqdm(total=N, position=0)

            m_kwargs = {}

            if(tr_controller == "inject"):
                m_kwargs["controller"] = InjectedController()
            elif(tr_controller == "transfer"):
                m_kwargs["controller"] = None
            elif(tr_controller == "homotopy_simple"):
                m_kwargs["controller"] = HomotopyController(False, eps=1e-2)
            elif(tr_controller == "homotopy_transfer"):   
                m_kwargs["controller"] = HomotopyController(True, 0.5, 1e-2) 

            for ep in range(N):
                model = EsnForecaster(
                    n_reservoir=500,
                    spectral_radius=0.9,
                    sparsity=0.1,
                    regularization='l2|noise',
                    lambda_r=5e-2,
                    noise_theta=1e-4,
                    in_activation='tanh',
                    random_state=args.seed,
                    **m_kwargs
                )

                model._fit(data[ep, ANC, :5000], controls[ANC] if (tr_controller != "transfer") else None, inspect = False, initialization_strategy = optimal_weights_generator(
                    verbose = 0,
                    range_generator=np.linspace,
                    steps = 400,
                    hidden_std = 0.5,
                    find_optimal_input = False,
                    thinning_step = 50,
                ))

                if("infer" in tr_type): break

                if(tr_controller == "transfer"): W_out_ = model.W_out_.copy()
                for i in range(20):
                    if(tr_controller == "transfer"): 
                        model._update_via_transfer_learning(data[0, i, :400], mu=1e-2)
                        p = model.predict(5000)
                        model.W_out_ = W_out_.copy()
                    else:
                        model.update(data[ep, i, :500], controls[i, :500] if (tr_controller != "transfer") else None, mode=UpdateModes.synchronization)
                        p = model.predict(5000, np.ones((5000,1)) * controls[i, 0] if (tr_controller != "transfer") else None)
                    for j in range(5):
                        w_errors[ep, i] += wasserstein_distance(data[ep, i,:,j], p[:,j])
                    w_errors[ep, i] /= 5

                for i in range(20):
                    if(tr_controller == "transfer"): 
                        model._update_via_transfer_learning(data[0, i, :400], mu=1e-2)
                        f_errors[ep, i] = forecasting(model, data[ep, i, :2000])
                        model.W_out_ = W_out_.copy()
                    else:
                        f_errors[ep, i] = forecasting(model, data[ep, i, :2000], controls[i, :2000] if (tr_controller != "transfer") else None)

                pbar.update(1)

                postfix = f"{tr_type.split('_')[-1]}_{tr_controller}"
                np.save(os.path.join(task_path, f"w_errors_{postfix}.npy"), w_errors)
                np.save(os.path.join(task_path, f"f_errors_{postfix}.npy"), f_errors)