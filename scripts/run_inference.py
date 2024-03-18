import os
os.environ["OMP_NUM_THREADS"] = str(1)

import numpy as np

from multiprocessing import Pool, cpu_count

import pandas as pd
import time

import sys
sys.path.append('/dipc/kstoreyf/muchisimocks/scripts')
import utils
import moment_network as mn
import mcmc
import scaler_custom as scl

import generate_emuPks as genP



def main():
    run_moment = False
    run_sbi = False
    run_emcee = True
    run_dynesty = False
    
    idxs_test = np.arange(15)

    data_mode = 'emuPk'
    if data_mode == 'emuPk':
        frac_train=0.4
        frac_val=0.1
        frac_test=0.5
        
    elif data_mode == 'muchisimocks':
        frac_train=0.70
        frac_val=0.15
        frac_test=0.15

    tag_data = '_2param'
    tag_errG = f'_boxsize500'
    tag_inf = '_'+data_mode + tag_data + tag_errG
    #tag_errG = f''
    n_tot = 2000
    
    rng = np.random.default_rng(42)
    
    dir_data = '../data/emuPks'
    theta, y, y_err, k, param_names = load_data_emuPk(dir_data, tag_data, tag_errG, rng=rng)
    if n_tot < theta.shape[0]:
        theta, y, y_err = subsample_data(theta, y, y_err, n_tot, rng=rng)
    n_tot, n_params = theta.shape
    bias_params = np.loadtxt(f'{dir_data}/bias_params.txt')

    random_ints = np.arange(n_tot)
    rng.shuffle(random_ints) #in-place

    idxs_train, idxs_val, idxs_test = utils.idxs_train_val_test(random_ints, 
                                   frac_train=frac_train, frac_val=frac_val, frac_test=frac_test)
    
    theta_train, theta_val, theta_test = utils.split_train_val_test(theta, idxs_train, idxs_val, idxs_test)
    y_train, y_val, y_test = utils.split_train_val_test(y, idxs_train, idxs_val, idxs_test)
    y_err_train, y_err_val, y_err_test = utils.split_train_val_test(y_err, idxs_train, idxs_val, idxs_test)
    
    ys_scaled = scale_y_data(y_train, y_val, y_test,
                           y_err_train=y_err_train, y_err_val=y_err_val, y_err_test=y_err_test,
                           return_scaler=True)
    y_train_scaled, y_val_scaled, y_test_scaled, \
               y_err_train_scaled, y_err_val_scaled, y_err_test_scaled, scaler = ys_scaled
    
    if run_moment:
        moment_network = mn.MomentNetwork(theta_train=theta_train, y_train=y_train_scaled, y_err_train=y_err_train_scaled,
                            theta_val=theta_val, y_val=y_val_scaled, y_err_val=y_err_val_scaled,
                            theta_test=theta_test, y_test=y_test_scaled, y_err_test=y_err_test_scaled,
                            tag_mn=tag_inf)
        moment_network.run(max_epochs_mean=1000, max_epochs_cov=1000)
        moment_network.evaluate_test_set()
        
    if run_emcee:
            
        for idx_test in idxs_test:
            y_data_unscaled = y_test[idx_test]
            y_data = y_test_scaled[idx_test]

            err_1p = 0.01*y_data_unscaled
            err_1p_scaled = scaler.scale_error(err_1p, y_data_unscaled)
            err_gaussian_scaled = y_err_test_scaled[idx_test]
            var = err_gaussian_scaled**2 + err_1p_scaled**2
            cov_inv = np.diag(1/var)
        
            emu, emu_bounds, emu_param_names = utils.load_emu()
            dict_bounds = {name: emu_bounds[emu_param_names.index(name)] for name in param_names}
            cosmo_params = utils.setup_cosmo_emu()
        
            mcmc.evaluate_emcee(idx_test, y_data, cov_inv, scaler,
                        emu, cosmo_params, bias_params, k,
                        dict_bounds, param_names, emu_param_names,
                        tag_inf=tag_inf)
        


def load_data_emuPk(dir_data, tag_data, tag_errG, rng=None):

    if rng is None:
        rng = np.random.default_rng(42)

    fn_emuPk = f'{dir_data}/emuPks{tag_data}.npy'
    fn_emuPk_params = f'{dir_data}/emuPks_params{tag_data}.txt'
    fn_emuk = f'{dir_data}/emuPks_k{tag_data}.txt'
    fn_emuPkerrG = f'{dir_data}/emuPks_errgaussian{tag_data}{tag_errG}.npy'

    Pk_noiseless = np.load(fn_emuPk)
    gaussian_error_pk = np.load(fn_emuPkerrG)
    theta = np.genfromtxt(fn_emuPk_params, delimiter=',', names=True)
    param_names = theta.dtype.names
    # from tuples to 2d array
    theta = np.array([list(tup) for tup in theta])
    k = np.genfromtxt(fn_emuk)
    
    # add noise
    Pk = rng.normal(Pk_noiseless, gaussian_error_pk)    
    
    print(theta.shape, Pk.shape, k.shape, gaussian_error_pk.shape)
    # should really do this mask just on training set, but for now have some bad test data for some reason
    mask = np.all(Pk>0, axis=0)
    Pk = Pk[:,mask]
    gaussian_error_pk = gaussian_error_pk[:,mask]
    k = k[mask]
    
    return theta, Pk, gaussian_error_pk, k, param_names


def subsample_data(theta, y, y_err, n_tot, rng=None):
    
    if rng is None:
        rng = np.random.default_rng(42)
        
    idx_select = rng.choice(np.arange(theta.shape[0]), n_tot, replace=False)
    theta = theta[idx_select]
    y = y[idx_select]
    y_err = y_err[idx_select]
    return theta, y, y_err
    


def scale_y_data(y_train, y_val, y_test,
                 y_err_train=None, y_err_val=None, y_err_test=None,
                 return_scaler=True):
    scaler = scl.Scaler()
    scaler.fit(y_train)
    y_train_scaled = scaler.scale(y_train)
    y_val_scaled = scaler.scale(y_val)
    y_test_scaled = scaler.scale(y_test)

    if y_err_train is not None:
        err_msg =  "If you're passing y_err_train, you should also pass y_err_val and y_err_test!"
        assert y_err_val is not None and y_err_test is not None, err_msg
        y_err_train_scaled = scaler.scale_error(y_err_train, y_train)
        y_err_val_scaled = scaler.scale_error(y_err_val, y_val)
        y_err_test_scaled = scaler.scale_error(y_err_test, y_test)
        if return_scaler:
            return y_train_scaled, y_val_scaled, y_test_scaled, \
                y_err_train_scaled, y_err_val_scaled, y_err_test_scaled, scaler
        else:
            return y_train_scaled, y_val_scaled, y_test_scaled, \
                y_err_train_scaled, y_err_val_scaled, y_err_test_scaled

    if return_scaler:
        return y_train_scaled, y_val_scaled, y_test_scaled, scaler
    else:
        return y_train_scaled, y_val_scaled, y_test_scaled




def get_test_data(idx_test):
    pk_data_unscaled = Pk_test_scaled[idx_test]
    pk_data = Pk_test_scaled[idx_test]

    err_1p = 0.01*pk_data_unscaled
    err_1p_scaled = scaler.scale_error(err_1p, pk_data_unscaled)
    err_gaussian_scaled = gaussian_error_pk_test_scaled[idx_test]
    var = err_gaussian_scaled**2 + err_1p_scaled**2
    cov_inv = np.diag(1/var)





if __name__=='__main__':
    main()