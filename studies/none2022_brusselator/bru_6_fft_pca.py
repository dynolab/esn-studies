import numpy as np
from matplotlib import pyplot as plt
from skesn.esn import EsnForecaster
from thequickmath.field import Space, map_to_2d_mesh

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
from skesn.cross_validation import ValidationBasedOnRollingForecastingOrigin
from skesn.weight_generators import optimal_weights_generator
from sklearn.decomposition import PCA

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
    x_ = t #np.linspace(-np.pi, np.pi, n_data.shape[0], endpoint=False)
    for ax in axes:
        ax.plot(x_, n_data[:,i], '-', linewidth=1.5, label="True")
        ax.plot(x_[:data_train.shape[0]], data_train[:,i], 'm--',linewidth=1.5, label="Train")
        ax.plot(x_[data_train.shape[0]:], data_predicted[:,i], '--',linewidth=1.5, label="Prediction")
        #ax.plot(x_, np.fft.ifft(a_hat_coeffs), '.-',linewidth=1)
        ax.grid()
        ax.legend()
        i += n_data.shape[1]//2
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
    plt.xlabel("u")
    plt.ylabel("v")
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

def a_coeff(a_hat):
    return 2*np.real(a_hat)
def b_coeff(a_hat):
    return -2*np.imag(a_hat)

def allab(x, coef_a, coef_b):
    result=0
    for i in range(len(coef_a)):
        result+= coef_a[i]*np.cos(i*x) + coef_b[i]*np.sin(i*x)
    return result

def get_spectrum(n_data):
    x_ = np.linspace(-np.pi, np.pi, n_data.shape[0], endpoint=False)
    a_hat_coeffs = (-1)**(np.arange(len(x_))) * np.fft.fft(n_data) / len(x_)
    a_coeffs = a_coeff(a_hat_coeffs[:int(len(x_)//2)])
    b_coeffs = b_coeff(a_hat_coeffs[:int(len(x_)//2)])
    return np.r_[a_coeffs, b_coeffs]

def get_from_spectrum(x_sh,coeffs):
    a_coeffs = coeffs[:coeffs.shape[0]//2]
    b_coeffs = coeffs[coeffs.shape[0]//2:]
    x_ = np.linspace(-np.pi, np.pi, x_sh, endpoint=False) #coeffs.shape[0]
    return allab(x_, a_coeffs, b_coeffs) - a_coeffs[0]/2 


if __name__ == '__main__':
    
    x, t, u_v_concat = _bru('2')
    #print(u_v_concat.shape)

    #находим спектры
    coeffs_data = np.zeros((0,512))
    a_hat_data = np.zeros((0,512))
    for i in range(u_v_concat.shape[0]):
        n_data = u_v_concat[i]
        coeffs = get_spectrum(n_data)
        coeffs_data = np.append(coeffs_data, [coeffs], axis=0)
    #print(coeffs_data.shape)

     #ESN
    rand=10
    np.random.seed(rand)
    esn_type = 'stand' #'stand' 'opt_param'
    if esn_type == 'stand':
        esn_bru_coeff = EsnForecaster(n_reservoir=1500,
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
        esn_bru_coeff = EsnForecaster(n_reservoir=1500,
                                   spectral_radius=0.95,
                                   sparsity=0.2,
                                   regularization='noise',
                                   lambda_r=0.005,
                                   in_activation='tanh',
                                   out_activation='identity',
                                   use_additive_noise_when_forecasting=True,
                                   random_state=rand,
                                   use_bias=True) 

    pca = PCA(n_components = 8)
    X_pca_reduced = pca.fit_transform(coeffs_data)

    #разделяем на обучающую и тестовую выборки
    time_train = 10000
    train_data_pca = np.array(X_pca_reduced[:time_train,:])
    #print(train_data_pca.shape)
    
    #standard fitting
    error = esn_bru_coeff.fit(train_data_pca,inspect=True)

    time_predict= coeffs_data.shape[0] - time_train
    prediction_coeffs = esn_bru_coeff.predict(time_predict,inspect=True).T 
    #print(prediction_coeffs.shape)

    X_pca_returned = pca.inverse_transform(prediction_coeffs.T)
    #print(X_pca_returned.shape)

    data_train_ret = pca.inverse_transform(train_data_pca)
    #print(data_train_ret.shape)

    #возвращаемся от спектров train
    data_train = np.zeros((0,512))
    for i in range(data_train_ret.shape[0]):
        data_train = np.append(data_train, [get_from_spectrum(x.shape[0]*2,data_train_ret[i])], axis=0)
    print(data_train.shape)

    #возвращаемся от спектров predict
    data_return = np.zeros((0,512))
    for i in range(X_pca_returned.shape[0]):
        data_return = np.append(data_return, [get_from_spectrum(x.shape[0]*2,X_pca_returned[i])], axis=0)
    print(data_return.shape)

    plot_data_train_pred(u_v_concat, data_train, data_return, 2, filename=f'1D_bru_images_note_11_04\\u_v_ot_t_esn_{esn_type}.png')

    plot_data_phase_traj(u_v_concat, data_train, data_return, 2, filename=f'1D_bru_images_note_11_04\\phase_tr_2_esn_{esn_type}.png') #, figsz=(4,4)

    t_pred = np.arange(20+0.01*time_train, 20+0.01*(time_train+time_predict), 0.01)

    plot_data_train_pred_mesh(x, t_pred, data_return[:,:256], data_return[:,256:], u_v_concat, figsz=(6,2.5), path='1D_bru_images_note_11_04', filename=f'pred_colormesh_esn_{esn_type}.png')

    import winsound
    frequency = 500  # Set Frequency To 2500 Hertz
    duration = 100  # Set Duration To 1000 ms == 1 second
    #winsound.Beep(frequency*10, duration)
    winsound.Beep(frequency, duration)

    print("Result is in the C:\\Users\\njuro\\Documents\\Диплом Магистратура\\Figures\\1D_bru_images_note_11_04")


    
