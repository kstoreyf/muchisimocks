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
import scaler_custom as scl

import generate_emuPks as genP



def main():
    run_moment = True
    run_sbi = False
    run_emcee = False
    run_dynesty = False
    
    # for likelihood methods only
    idxs_obs = np.arange(1)

    # train test split, and number of rlzs per cosmo
    n_train = 8000
    n_val = 1000
    #n_train = 100
    #n_val = 250
    n_test = 1000

    ### Load data
    data_mode = 'muchisimocksPk'
    #data_mode = 'emuPk'
    assert data_mode in ['emuPk', 'muchisimocksPk']
    
    if data_mode == 'emuPk':
        #n_rlzs_per_cosmo = 9
        n_rlzs_per_cosmo = 1
        #tag_emuPk = '_5param'
        tag_emuPk = '_2param'
        tag_errG = f'_boxsize500'
        tag_datagen = f'{tag_emuPk}{tag_errG}_nrlzs{n_rlzs_per_cosmo}'
        
        theta, y, y_err, k, param_names, bias_params, random_ints = load_data_emuPk(tag_emuPk, 
                                                        tag_errG, tag_datagen,
                                                        n_rlzs_per_cosmo=n_rlzs_per_cosmo)
    elif data_mode == 'muchisimocksPk':
        tag_mocks = '_p5_n10000'
        tag_pk = '_b1000'
        tag_datagen = f'{tag_mocks}{tag_pk}'
        theta, y, y_err, k, param_names, bias_params, random_ints = load_data_muchisimocksPk(tag_mocks,
                                                                                             tag_pk)
    elif data_mode == 'muchisimocks3D':
        tag_mocks = '_p5_n10000'
        tag_datagen = f'{tag_mocks}'
        theta, y, y_err, k, param_names, bias_params, random_ints = load_data_muchisimocks3D(tag_mocks)
               
    tag_data = '_'+data_mode + tag_datagen
    print(theta.shape, y.shape, y_err.shape, random_ints.shape)

    ### Train-val-test split
    if data_mode == 'emuPk':
        # leave this fixed so we don't mix the sets; then we'll subsample
        # frac_train=0.85
        # frac_val=0.05
        # frac_test=0.1
        frac_train=0.8
        frac_val=0.1
        frac_test=0.1
    elif data_mode == 'muchisimocksPk':
        frac_train=0.8
        frac_val=0.1
        frac_test=0.1
        
    idxs_train, idxs_val, idxs_test = utils.idxs_train_val_test(random_ints, 
                                   frac_train=frac_train, frac_val=frac_val, frac_test=frac_test)
    
    idxs_train = idxs_train[:n_train]
    idxs_val = idxs_val[:n_val]
    idxs_test = idxs_test[:n_test]
    
    n_emuPk = len(random_ints)
    idxs_train_orig = idxs_train.copy()
    idxs_train = []
    if data_mode == 'emuPk':
        for n_rlz in range(n_rlzs_per_cosmo):
            idxs_train.extend(idxs_train_orig + n_rlz*n_emuPk)
    elif 'muchisimocks' in data_mode:
        idxs_train.extend(idxs_train_orig)
    idxs_train = np.array(idxs_train)
    print(idxs_train.shape)
            
    theta_train, theta_val, theta_test = utils.split_train_val_test(theta, idxs_train, idxs_val, idxs_test)
    y_train, y_val, y_test = utils.split_train_val_test(y, idxs_train, idxs_val, idxs_test)
    y_err_train, y_err_val, y_err_test = utils.split_train_val_test(y_err, idxs_train, idxs_val, idxs_test)
    print(theta_train.shape, theta_val.shape, theta_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)
    print(y_err_train.shape, y_err_val.shape, y_err_test.shape)
    
    ### Scale data
    ys_scaled = scale_y_data(y_train, y_val, y_test,
                           y_err_train=y_err_train, y_err_val=y_err_val, y_err_test=y_err_test,
                           return_scaler=True)
    y_train_scaled, y_val_scaled, y_test_scaled, \
               y_err_train_scaled, y_err_val_scaled, y_err_test_scaled, scaler = ys_scaled
    
    ### Run inference
    if run_moment:
        #tag_inf = f'{tag_data}_ntrain{n_train}_scalecovminmax'
        #tag_inf = f'{tag_data}_ntrain{n_train}_eigs'
        tag_inf = f'{tag_data}_ntrain{n_train}_direct'
        moment_network = mn.MomentNetwork(theta_train=theta_train, y_train=y_train_scaled, y_err_train=y_err_train_scaled,
                            theta_val=theta_val, y_val=y_val_scaled, y_err_val=y_err_val_scaled,
                            theta_test=theta_test, y_test=y_test_scaled, y_err_test=y_err_test_scaled,
                            tag_mn=tag_inf)
        #moment_network.run(max_epochs_mean=5, max_epochs_cov=5)
        moment_network.run(max_epochs_mean=2000, max_epochs_cov=2000)
        moment_network.evaluate_test_set()
        print(f"Saved results to {moment_network.dir_mn}")
        
    if run_emcee:
        
        import mcmc
        emu, emu_bounds, emu_param_names = utils.load_emu()
        dict_bounds = {name: emu_bounds[emu_param_names.index(name)] for name in param_names}
        cosmo_params = utils.setup_cosmo_emu()   
            
        for idx_obs in idxs_obs:
            
            y_data_unscaled = y_test[idx_obs]
            y_data = y_test_scaled[idx_obs]

            err_1p = 0.01*y_data_unscaled
            err_1p_scaled = scaler.scale_error(err_1p, y_data_unscaled)
            err_gaussian_scaled = y_err_test_scaled[idx_obs]
            var = err_gaussian_scaled**2 + err_1p_scaled**2
            cov_inv = np.diag(1/var)
        
            mcmc.evaluate_emcee(idx_obs, y_data, cov_inv, scaler,
                        emu, cosmo_params, bias_params, k,
                        dict_bounds, param_names, emu_param_names,
                        tag_inf=tag_inf)
            
    if run_dynesty:
        
        import mcmc
        emu, emu_bounds, emu_param_names = utils.load_emu()
        dict_bounds = {name: emu_bounds[emu_param_names.index(name)] for name in param_names}
        cosmo_params = utils.setup_cosmo_emu()
                        
        for idx_obs in idxs_obs:
            
            y_data_unscaled = y_test[idx_obs]
            y_data = y_test_scaled[idx_obs]

            err_1p = 0.01*y_data_unscaled
            err_1p_scaled = scaler.scale_error(err_1p, y_data_unscaled)
            err_gaussian_scaled = y_err_test_scaled[idx_obs]
            var = err_gaussian_scaled**2 + err_1p_scaled**2
            cov_inv = np.diag(1/var)
        
            mcmc.evaluate_dynesty(idx_obs, y_data, cov_inv, scaler,
                        emu, cosmo_params, bias_params, k,
                        dict_bounds, param_names, emu_param_names,
                        tag_inf=tag_inf)
        

def load_data_muchisimocksPk(tag_mocks, tag_pk):       
     
    dir_params = '../data/params'
    fn_params = f'{dir_params}/params_lh{tag_mocks}.txt'
    params_df = pd.read_csv(fn_params, index_col=0)
    param_names = params_df.columns.tolist()
    idxs_LH = params_df.index.tolist()

    dir_pks = f'../data/pks_mlib/pks{tag_mocks}{tag_pk}'
    fn_bias_vector = f'{dir_pks}/bias_params.txt'
    bias_vector = np.loadtxt(fn_bias_vector)

    fn_rands = f'{dir_params}/randints{tag_mocks}.npy'
    random_ints = np.load(fn_rands)

    theta, Pk, gaussian_error_pk = [], [], []
    for idx_LH in idxs_LH:
        fn_pk = f'{dir_pks}/pk_{idx_LH}.npy'
        pk_obj = np.load(fn_pk, allow_pickle=True).item()
        Pk.append(pk_obj['pk'])
        gaussian_error_pk.append(pk_obj['pk_gaussian_error'])
        param_vals = params_df.loc[idx_LH].values
        theta.append(param_vals)

    Pk = np.array(Pk)
    theta = np.array(theta)
    gaussian_error_pk = np.array(gaussian_error_pk)
    k = pk_obj['k'] # all ks should be same so just grab one

    # TODO do i want / need this here?
    mask = np.all(Pk>0, axis=0)
    print(f"{np.sum(np.any(Pk<=0, axis=1))}/{Pk.shape[0]} have at least one non-positive Pk value")
    print(f"Masking columns {np.where(~mask)[0]}")
    Pk = Pk[:,mask]
    gaussian_error_pk = gaussian_error_pk[:,mask]
    k = k[mask]
    
    return theta, Pk, gaussian_error_pk, k, param_names, bias_vector, random_ints

        
def load_data_emuPk(tag_emuPk, tag_errG, tag_datagen, 
                    n_rlzs_per_cosmo=1, return_noiseless=False):

    dir_data = '../data/emuPks'
    fn_emuPk = f'{dir_data}/emuPks{tag_emuPk}.npy'
    fn_emuPkerrG = f'{dir_data}/emuPks_errgaussian{tag_emuPk}{tag_errG}.npy'
    fn_emuPk_noisy = f'{dir_data}/emuPks_noisy{tag_datagen}.npy'
    fn_emuPk_params = f'{dir_data}/emuPks_params{tag_emuPk}.txt'
    fn_emuk = f'{dir_data}/emuPks_k{tag_emuPk}.txt'
    fn_bias_vector = f'{dir_data}/bias_params.txt'
    fn_rands = f'{dir_data}/randints{tag_emuPk}.npy'
        
    Pk = np.load(fn_emuPk_noisy, allow_pickle=True)   
    Pk_noiseless = np.load(fn_emuPk, allow_pickle=True)
    k = np.genfromtxt(fn_emuk)
    bias_vector = np.loadtxt(fn_bias_vector)
    random_ints = np.load(fn_rands, allow_pickle=True)

    theta_noiseless = np.genfromtxt(fn_emuPk_params, delimiter=',', names=True)
    print("theta_noiseless", theta_noiseless.shape)
    param_names = theta_noiseless.dtype.names
    theta_noiseless = np.array([list(tup) for tup in theta_noiseless]) # from tuples to 2d array
    gaussian_error_pk_noiseless = np.load(fn_emuPkerrG, allow_pickle=True)
    
    theta = utils.repeat_arr_rlzs(theta_noiseless, n_rlzs=n_rlzs_per_cosmo)
    gaussian_error_pk = utils.repeat_arr_rlzs(gaussian_error_pk_noiseless, n_rlzs=n_rlzs_per_cosmo)
    
    mask = np.all(Pk>0, axis=0)
    Pk = Pk[:,mask]
    Pk_noiseless = Pk_noiseless[:,mask]
    gaussian_error_pk = gaussian_error_pk[:,mask]
    gaussian_error_pk_noiseless = gaussian_error_pk_noiseless[:,mask]
    k = k[mask]
    
    if return_noiseless:
        return theta, Pk, gaussian_error_pk, k, param_names, bias_vector, random_ints, \
               theta_noiseless, Pk_noiseless, gaussian_error_pk_noiseless
    
    else:
        return theta, Pk, gaussian_error_pk, k, param_names, bias_vector, random_ints


# def load_data_emuPk(dir_data, tag_datagen, tag_errG, rng=None,
#                     n_rlzs_per_cosmo=1):

#     if rng is None:
#         rng = np.random.default_rng(42)

#     fn_emuPk = f'{dir_data}/emuPks{tag_datagen}.npy'
#     fn_emuPk_params = f'{dir_data}/emuPks_params{tag_datagen}.txt'
#     fn_emuk = f'{dir_data}/emuPks_k{tag_datagen}.txt'
#     fn_emuPkerrG = f'{dir_data}/emuPks_errgaussian{tag_datagen}{tag_errG}.npy'

#     Pk_noiseless = np.load(fn_emuPk)
#     gaussian_error_pk = np.load(fn_emuPkerrG)
#     theta = np.genfromtxt(fn_emuPk_params, delimiter=',', names=True)
#     param_names = theta.dtype.names
#     # from tuples to 2d array
#     theta = np.array([list(tup) for tup in theta])
#     k = np.genfromtxt(fn_emuk)
#     print(theta.shape, Pk_noiseless.shape, k.shape, gaussian_error_pk.shape)
    
#     # add noise
#     #Pk = rng.normal(Pk_noiseless, gaussian_error_pk)    
#     # generate more shape- doing this hackily for now to keep same realizations as mcmc for first set
#     #Pk = np.empty(Pk_noiseless.shape)
#     Pk = rng.normal(Pk_noiseless, gaussian_error_pk)    
#     for _ in range(n_rlzs_per_cosmo-1):
#         # i think first set should be equiv to orig?
#         Pk_wnoise = rng.normal(Pk_noiseless, gaussian_error_pk)    
#         Pk = np.vstack((Pk, Pk_wnoise))
#     # repeat theta, err arrs
#     theta = np.repeat(theta, n_rlzs_per_cosmo, axis=0)
#     gaussian_error_pk = np.repeat(gaussian_error_pk, n_rlzs_per_cosmo, axis=0)
    
#     print('After rlzs:', theta.shape, Pk.shape, k.shape, gaussian_error_pk.shape)
#     # should really do this mask just on training set, but for now have some bad test data for some reason
#     mask = np.all(Pk>0, axis=0)
#     Pk = Pk[:,mask]
#     gaussian_error_pk = gaussian_error_pk[:,mask]
#     k = k[mask]
    
#     return theta, Pk, gaussian_error_pk, k, param_names



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




# def get_test_data(idx_test):
#     pk_data_unscaled = Pk_test_scaled[idx_test]
#     pk_data = Pk_test_scaled[idx_test]

#     err_1p = 0.01*pk_data_unscaled
#     err_1p_scaled = scaler.scale_error(err_1p, pk_data_unscaled)
#     err_gaussian_scaled = gaussian_error_pk_test_scaled[idx_test]
#     var = err_gaussian_scaled**2 + err_1p_scaled**2
#     cov_inv = np.diag(1/var)





if __name__=='__main__':
    main()