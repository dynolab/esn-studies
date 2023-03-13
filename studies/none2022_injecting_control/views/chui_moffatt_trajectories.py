import numpy as np
from matplotlib import pyplot as plt
from skesn.esn import EsnForecaster, UpdateModes

import argparse
import os
from tqdm import tqdm
from matplotlib.gridspec import GridSpec

from skesn.weight_generators import optimal_weights_generator
from skesn.esn_controllers import *
from skesn.data_preprocess import ToNormalConverter

from comsdk.research import Research

def get_maximas(v):
    left_difs = v[1:-1] - v[:-2]
    right_difs = v[1:-1] - v[2:]
    dif_prods = left_difs * right_difs
    return np.where((left_difs > 0) * (right_difs > 0))[0]+1

if __name__ == "__main__":
    task = 0
    res_id = "CTL"
    res = Research.open(res_id)
    task_path = res.get_task_path(task)

    parser = argparse.ArgumentParser(description="Experiments with chui model")
    parser.add_argument("--type", choices=["preproc_orig", "preproc_proc", "preproc_log", "infer_orig", "infer_proc", "infer_log", "all"], default="all", help="Type of the experiment")
    parser.add_argument("--controller", choices=["inject", "transfer", "homotopy_simple", "homotopy_transfer", "all"], default="all", help="Type of controller")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--xi", type=int, default=32, help="Value of the analyzed xi")
    parser.add_argument("--regenerate_data", action='store_true', help="Regenerate training data")
    parser.add_argument("--save_folder", type=str, default="./", help="Folder to save figures")

    args = parser.parse_args()

    figfolder = args.save_folder

    xi_idx = args.xi / 2 - 1
    xi_idx_i = int(xi_idx)

    time = np.arange(0, 200, 0.04)
    data_orig = np.load(os.path.join(task_path, "data_chui.npy"))
    data = data_orig.copy()

    if(args.type == "all"):
        tr_types = ["preproc_orig", "preproc_proc", "preproc_log", "infer_orig", "infer_proc", "infer_log"]
    else: tr_types = [args.type]
    if(args.controller == "all"):
        tr_controllers = ["inject", "transfer", "homotopy_simple", "homotopy_transfer"]
    else: tr_controllers = [args.controller]

    for tr_type in tr_types:
        for tr_controller in tr_controllers:
            print("Trajectories for (%s, %s)..." % (tr_type, tr_controller))

            #### DRAW ORIGINAL DATA ####
            if("preproc" in tr_type):
                plt.figure(figsize=(12,5))
                for i in range(5):
                    plt.subplot(5, 1, i+1)
                    plt.plot(time[:2500], data[1, xi_idx_i, :2500, i])
                    plt.ylabel(f"${'xyzuv'[i]}$")
                    if(i == 4): plt.xlabel("$t$")
                    else: plt.xticks([])
                plt.tight_layout()
                plt.savefig(os.path.join(figfolder, "original_timeseries.png"), dpi=200)

                plt.figure(figsize=(20,3))
                for i in range(5):
                    plt.subplot(1, 5, i+1)
                    plt.hist(data[:4, xi_idx_i, :, i].flatten(), 100, density=True)
                    plt.xlabel(f"${'xyzuv'[i]}$")
                plt.tight_layout()
                plt.savefig(os.path.join(figfolder, "original_distribution.png"), dpi=200)
            ############################

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

            
            if("preproc_proc" in tr_type): 
                # scaler = ToNormalConverter().fit(data[:4, 15:, :, 1].reshape(-1))

                plt.figure(figsize=(12,5))
                t = np.linspace(-5, 5, 100)
                plt.plot(t, scaler.data_to_uni_(t))
                plt.xlabel("$x$")
                plt.ylabel("$F_{\\xi_1}(x)$")
                plt.tight_layout()
                plt.savefig(os.path.join(figfolder, "preprocessed_data2unif.png"), dpi=200)

                plt.figure(figsize=(12,5))
                t = np.linspace(-5, 5, 100)
                plt.plot(t, scaler.norm_to_uni_(t))
                plt.xlabel("$x$")
                plt.ylabel("$F_{\\xi_2}(x)$")
                plt.tight_layout()
                plt.savefig(os.path.join(figfolder, "preprocessed_unif2norm.png"), dpi=200)

            #### DRAW PREPROCESSED DATA ####
            if("preproc" in tr_type):
                plt.figure(figsize=(12,5))
                for i in range(5):
                    plt.subplot(5, 1, i+1)
                    plt.plot(time[:2500], data[0, xi_idx_i, :2500, i])
                    plt.ylabel(f"${'xyzuv'[i]}$")
                    if(i == 4): plt.xlabel("$t$")
                    else: plt.xticks([])
                plt.tight_layout()
                plt.savefig(os.path.join(figfolder, "preprocessed_timeseries.png"), dpi=200)

                plt.figure(figsize=(20,3))
                for i in range(5):
                    plt.subplot(1, 5, i+1)
                    plt.hist(data[:4, xi_idx_i, :, i].flatten(), 100, density=True)
                    plt.xlabel(f"${'xyzuv'[i]}$")
                plt.tight_layout()
                plt.savefig(os.path.join(figfolder, "preprocessed_distribution.png"), dpi=200)

                continue
            ############################

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

            postfix = f"{tr_type.split('_')[-1]}_{tr_controller}"
            w_errors = np.load(os.path.join(task_path, f"w_errors_{postfix}.npy"))
            f_errors = np.load(os.path.join(task_path, f"f_errors_{postfix}.npy"))

            # DRAW PREDICTED TIMESERIES
            w_error = np.median(w_errors, 0)
            f_error = np.median(f_errors, 0)
            w_std = np.std(w_errors, 0)
            f_std = np.std(f_errors, 0)
            w_maxs = np.sort(w_errors, axis=0)[-4]
            f_maxs = np.sort(f_errors, axis=0)[-4]

            plt.figure(figsize=(12,5))

            gs = GridSpec(10, 2, figure=plt.gcf())
            axs = [plt.gcf().add_subplot(gs[i*2:i*2+2, 0]) for i in range(5)]
            ax2 = plt.gcf().add_subplot(gs[:5, 1])
            ax3 = plt.gcf().add_subplot(gs[5:, 1])
            ax2.set_xticks([])

            if(tr_controller == "transfer"): 
                W_out_ = model.W_out_.copy()
                model._update_via_transfer_learning(data[0, xi_idx_i, :400], mu=1e-2)
            model.update(data[0, xi_idx_i, :400], controls[xi_idx_i] if (tr_controller != "transfer") else None, mode=UpdateModes.synchronization)
            output = model.predict(600, controls[xi_idx_i] if (tr_controller != "transfer") else None)
            if(tr_controller == "transfer"): model.W_out_ = W_out_.copy()

            D = data_orig[0, xi_idx_i]
            O = output.copy()

            for k in range(5):
                if(not (k == 1 and ("preproc_proc" in tr_type or "preproc_log" in tr_type))): 
                    O[:, k] = O[:, k] * STD[k] + MED[k]
                if(k == 1 and "_proc" in tr_type): O[:, k] = scaler.inverse_transform(O[:, k])
                if(k == 1 and "_log" in tr_type): O[:, k] = np.exp(O[:, k] * STD1 + MED1)
                    
            for i in range(5):
                if(i < 4): axs[i].set_xticks([])
                axs[i].plot(time[:401], D[:401,i], label="Synchronization")
                axs[i].plot(time[400:1000], O[:,i], label="Prediction")
                axs[i].plot(time[400:1000], D[400:1000,i], "--", label="Target")
                axs[i].set_ylabel("$%s$" % ("xyzuv"[i], ), rotation=0)

            axs[0].set_ylim(-4, 4)
            axs[1].set_ylim(-1, 4)
            axs[0].legend(loc = (0.02, 1.1), ncol=3)

            axs[4].set_xlabel("$t$")
            ax3.set_xlabel("$\\xi$")
            ax2.set_ylabel("Wasserstein m.")
            ax3.set_ylabel("Forecasting m.")

            rhos = np.linspace(1,40,20)
            ax3.set_xticks(list(range(0, 44, 4)))
            for i in range(2):
                ax = [ax2, ax3][i]
                E = [w_error, f_error][i]
                M = [w_maxs, f_maxs][i]
                ax.semilogy(rhos, E)
                ax.semilogy(rhos[ANC], E[ANC],"o",color="red")
                ax.semilogy(rhos[[xi_idx_i]], E[[xi_idx_i]],"o",color="green")
                ax.fill_between(rhos, 0, M, alpha = 0.25)

            plt.tight_layout()
            plt.savefig(os.path.join(figfolder, "predicted_timeseries.png"), dpi=200)

            # DRAW LAMBDA FIGURE
            plt.figure(figsize=(5,4))
            if(tr_controller == "transfer"): 
                W_out_ = model.W_out_.copy()
                model._update_via_transfer_learning(data[0, xi_idx_i, :400], mu=1e-2)
            model.update(data[0, xi_idx_i, :400], controls[xi_idx_i] if (tr_controller != "transfer") else None, mode=UpdateModes.synchronization)
            predict = model.predict(5000, np.array([controls[xi_idx_i, 0, 0]] * 5000) if (tr_controller != "transfer") else None)
            if(tr_controller == "transfer"): model.W_out_ = W_out_.copy()

            D = data_orig[0, xi_idx_i]
            O = predict.copy()

            for k in range(5):
                if(not (k == 1 and ("preproc_proc" in tr_type or "preproc_log" in tr_type))): 
                    O[:, k] = O[:, k] * STD[k] + MED[k]
                if(k == 1 and "_proc" in tr_type): O[:, k] = scaler.inverse_transform(O[:, k])
                if(k == 1 and "_log" in tr_type): O[:, k] = np.exp(O[:, k] * STD1 + MED1)

            tv = D[:, -1]
            pv = O[:, -1]

            idx = get_maximas(tv)
            plt.scatter(tv[idx[:-1]], tv[idx[1:]], alpha=0.5, label="true")

            idx = get_maximas(pv)
            plt.scatter(pv[idx[:-1]], pv[idx[1:]], alpha=0.5, label="predict")
            plt.xlabel("$v_{i}$")
            plt.ylabel("$v_{i+1}$")
            plt.tight_layout()
            plt.savefig(os.path.join(figfolder, "predicted_lambda.png"), dpi=200)

            # DRAW PREDICTED DATA DISTRIBUTION
            plt.figure(figsize=(20,3))
            for i in range(5):
                plt.subplot(1, 5, i+1)
                plt.hist(D[:, i], 100, fc=(0, 0, 1, 0.5), density=True, label="true")
                plt.hist(O[:1000, i], 100, fc=(1, 0, 0, 0.5), density=True, label="predict")
                if(i == 0): plt.legend()
                plt.xlabel("xyzuv"[i])
            plt.tight_layout()
            plt.savefig(os.path.join(figfolder, "predicted_distribution.png"), dpi=200)

