import numpy as np
from matplotlib import pyplot as plt
from skesn.esn import EsnForecaster
from thequickmath.field import Space, map_to_2d_mesh

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
from skesn.cross_validation import ValidationBasedOnRollingForecastingOrigin
from skesn.weight_generators import optimal_weights_generator

from sys import argv

def _bru(data_number):
    filename = 'C:\\Users\\njuro\\Documents\\restools\\Researches\\2022-07-26-predicting-brusselator-via-esn\\1-Dataset_from_Calum_1D_Brusselator\\brusselator1DB_%s.npz'  %data_number
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

    #объединение u и v в один массив
    u_v_concat = np.zeros((u.shape[0], u.shape[1]+v_new.shape[1]))
    u_v_concat[:,:256] = u
    u_v_concat[:,256:] = v_new
    
    return x, t, u_v_concat

def plot_data_train_pred(n_data, data_train, data_predicted, i, filename=None):
    fig, axes = plt.subplots(2,1, figsize=(15,5))
    x_ = t[::t_miss] #np.linspace(-np.pi, np.pi, n_data.shape[0], endpoint=False)
    for ax in axes:
        ax.plot(x_, n_data[:,i], '-', linewidth=1.5, label="True")
        ax.plot(x_[:data_train.shape[0]], data_train[:,i], 'm--',linewidth=1.5, label="Train")
        ax.plot(x_[data_train.shape[0]:], data_predicted[:,i], '--',linewidth=1.5, label="Prediction")
        #ax.plot(x_, np.fft.ifft(a_hat_coeffs), '.-',linewidth=1)
        ax.grid()
        ax.legend()
        i += n_data.shape[1]//2
    axes[0].set_ylabel(r'$u$', fontsize=14)
    axes[1].set_ylabel(r'$v$', fontsize=14)
    plt.xlabel(r'$t$', fontsize=14)
    if filename is not None:
        plt.savefig(f"C:\\Users\\njuro\\Documents\\Диплом Магистратура\\Figures\\{filename}")
 #   plt.show()

def plot_data_phase_traj(n_data, data_train, data_predicted, i, figsz=(7,7), filename=None):
    fig, ax = plt.subplots( figsize=figsz)
    u = n_data[:,:256]
    v = n_data[:,256:]
    plt.plot(u[:,i],v[:,i], '-',  label="True")
    plt.plot(data_train[:,i],data_train[:,256+i], 'm--', label="Train")
    plt.plot(data_predicted[:,i],data_predicted[:,256+i], '-', label="Prediction")
    plt.xlabel(r'$u$', fontsize=14)
    plt.ylabel(r'$v$', fontsize=14)
    plt.legend()
    if filename is not None:
        plt.savefig(f"C:\\Users\\njuro\\Documents\\Диплом Магистратура\\Figures\\{filename}")
 #   plt.show()

def plot_data_train_pred_mesh(x, t_pred, u_pred, v_pred, u_v_concat, figsz=(10,4), path='folder', filename=None):
    #предсказанное colormesh u
    fig, ax = plt.subplots(1,2, figsize=figsz)
    u = u_v_concat[:,:256]
    v = u_v_concat[:,256:]
    p1 = ax[0].pcolormesh(x, t_pred[:u_pred.shape[0]-1], u[time_train:-1,:], shading='nearest',cmap='plasma')
    p2 = ax[1].pcolormesh(x, t_pred[:u_pred.shape[0]-1], u_pred[:-1,:], shading='nearest',cmap='plasma')
    ax[0].set_xlabel(r'$x$')
    ax[1].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$t$')
    fig.colorbar(p1, ax=ax[0])
    fig.colorbar(p2, ax=ax[1])
    plt.suptitle("u and u_pred")
    plt.tight_layout()
    #plt.ylim([175,200])
    if filename is not None:
        plt.savefig(f"C:\\Users\\njuro\\Documents\\Диплом Магистратура\\Figures\\{path}\\u_{filename}")
 #   plt.show()

    #предсказанное colormesh v
    fig, ax = plt.subplots(1,2, figsize=figsz)
    p1 = ax[0].pcolormesh(x, t_pred[:v_pred.shape[0]-1], v[time_train:-1,:], shading='nearest',cmap='plasma')
    p2 = ax[1].pcolormesh(x, t_pred[:v_pred.shape[0]-1], v_pred[:-1,:], shading='nearest',cmap='plasma')
    ax[0].set_xlabel(r'$x$')
    ax[1].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$t$')
    fig.colorbar(p1, ax=ax[0])
    fig.colorbar(p2, ax=ax[1])
    plt.suptitle("v and v_pred")
    plt.tight_layout()
    #plt.ylim([175,200])
    if filename is not None:
        plt.savefig(f"C:\\Users\\njuro\\Documents\\Диплом Магистратура\\Figures\\{path}\\v_{filename}")
 #   plt.show()



if __name__ == '__main__':
    #t_miss sets time step: step = 0.01 * t_miss
    if len(argv) >1:
        t_miss = int(argv[1])
    else:
        t_miss=1
    print(" t_miss ", t_miss, "\n step ", t_miss*0.01)
    x, t, u_v_concat = _bru('2')
    u_v_concat = u_v_concat[::t_miss,:]
    #print(u_v_concat.shape)

     #ESN
    rand=10
    np.random.seed(rand)
    esn_type = 'opt_param' #'stand' 'opt_param'
    if esn_type == 'stand':
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
    elif esn_type == 'opt_param':
        esn_bru_uv = EsnForecaster(n_reservoir=1500,
                                   spectral_radius=0.95,
                                   sparsity=0.2,
                                   regularization='noise',
                                   lambda_r=0.005,
                                   in_activation='tanh',
                                   out_activation='identity',
                                   use_additive_noise_when_forecasting=True,
                                   random_state=rand,
                                   use_bias=True) 
    time_train = 10000 // t_miss
    train_data_uv = np.array(u_v_concat[:time_train,:])
    print("train_data_uv.shape ", train_data_uv.shape)

    #standard fitting
    error = esn_bru_uv.fit(train_data_uv,inspect=True)

    time_predict=u_v_concat.shape[0] - time_train
    prediction_uv = esn_bru_uv.predict(time_predict,inspect=True).T 
    #print(prediction_uv.shape)

    #разделяем предсказанное на u и v
    u_pred = prediction_uv.T[:,:256]
    v_pred = prediction_uv.T[:,256:]
    #print(u_pred.shape)
    #print(v_pred.shape)

    t_step = t_miss * 0.01
    t_pred = np.arange(20+t_step*time_train, 20+t_step*(time_train+time_predict), t_step)
    #print(t_pred.shape)

    plot_data_train_pred(u_v_concat, train_data_uv, prediction_uv.T, 2, filename=f'1D_bru_images_note_10_24\\u_v_ot_t_esn_{esn_type}_miss_{t_miss}.png')

    plot_data_phase_traj(u_v_concat, train_data_uv, prediction_uv.T, 2, filename=f'1D_bru_images_note_10_24\\phase_tr_2_esn_{esn_type}_miss_{t_miss}.png') #, figsz=(4,4)

    plot_data_train_pred_mesh(x, t_pred, u_pred, v_pred, u_v_concat, figsz=(6,2.5), path='1D_bru_images_note_10_24', filename=f'pred_colormesh_esn_{esn_type}_miss_{t_miss}.png')

    import winsound
    frequency = 500  # Set Frequency To 2500 Hertz
    duration = 100  # Set Duration To 1000 ms == 1 second
    #winsound.Beep(frequency*10, duration)
    winsound.Beep(frequency, duration)

    print("Result is in the C:\\Users\\njuro\\Documents\\Диплом Магистратура\\Figures\\1D_bru_images_note_10_24")


    
