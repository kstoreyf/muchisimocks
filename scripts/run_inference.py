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
import moment_network_dataloader as mndl
import scaler_custom as scl
import data_loader

import generate_emuPks as genP


def main():
    #run_3D_inference()
    #run_pk_inference()
    test_pk_inference()


def run_3D_inference():
    run_moment = True
    run_sbi = False
    
    n_threads = 1

    # train test split
    n_train = 6500
    n_val = 1000
    #n_train = 100
    #n_val = 250
    n_test = 1000

    ### Load data
    data_mode = 'muchisimocks3D'
    #data_mode = 'emuPk'
    assert data_mode in ['muchisimocks3D']
    
    if data_mode == 'muchisimocks3D':
        tag_mocks = '_p5_n10000'
        tag_datagen = f'{tag_mocks}'
        dataset_train, dataset_val, dataset_test, param_names = data_loader.load_data_muchisimocks3D(tag_mocks,
                                                                                                     n_train, n_val, n_test,
                                                                                                     n_threads=n_threads)
               
    tag_data = '_'+data_mode + tag_datagen

    # TODO figure out what to do with scaling here!!
    ### Scale data
    # ys_scaled = scale_y_data(y_train, y_val, y_test,
    #                        y_err_train=y_err_train, y_err_val=y_err_val, y_err_test=y_err_test,
    #                        return_scaler=True)
    # y_train_scaled, y_val_scaled, y_test_scaled, \
    #            y_err_train_scaled, y_err_val_scaled, y_err_test_scaled, scaler = ys_scaled
    
    ### Run inference
    if run_moment:
        #tag_inf = f'{tag_data}_ntrain{n_train}_scalecovminmax'
        #tag_inf = f'{tag_data}_ntrain{n_train}_eigs'
        tag_inf = f'{tag_data}_ntrain{n_train}_direct'
        moment_network = mndl.MomentNetworkDL(dataset_train, dataset_val, dataset_test,
                            tag_mn=tag_inf)
        #moment_network.run(max_epochs_mean=5, max_epochs_cov=5)
        moment_network.run(max_epochs_mean=2000, max_epochs_cov=2000)
        moment_network.evaluate_test_set()
        print(f"Saved results to {moment_network.dir_mn}")
        


def run_pk_inference():
    run_moment = True
    run_sbi = False
    run_emcee = False
    run_dynesty = False
    
    # for likelihood methods only
    idxs_obs = np.arange(1)

    # train test split, and number of rlzs per cosmo
    n_train = 8000
    n_val = 1000
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
        #tag_pk = '_b1000'
        #mode_bias_vector = 'single'
        tag_pk = '_biaszen_p4_n10000'
        mode_bias_vector = 'LH'
        tag_datagen = f'{tag_mocks}{tag_pk}'
        theta, y, y_err, k, param_names, bias_params, random_ints = load_data_muchisimocksPk(tag_mocks,
                                                                                             tag_pk,
                                                                                             mode_bias_vector=mode_bias_vector
                                                                                             )
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
        
    print(len(random_ints))
    idxs_train, idxs_val, idxs_test = utils.idxs_train_val_test(random_ints, 
                                   frac_train=frac_train, frac_val=frac_val, frac_test=frac_test,
                                   N_tot=10000)
    print(len(idxs_train), len(idxs_val), len(idxs_test))
    
    # CAUTION the subsamples will be very overlapping! this is ok for directly checking
    # for size dependence (less variability), but in a robust trend test would want to
    # do multiple random subsamples of a given size.
    # IF WANT TO DO RANDOM, make sure to set seed / do something smart to get same 
    # subsample back for testing!
    idxs_train = idxs_train[:n_train]
    idxs_test = idxs_test[:n_test]
    idxs_val = idxs_val[:n_val]
    
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
    
    # NOW DOING IN MOMENT_NETWORK
    # will have to do for each inf method
    ### Scale data
    # ys_scaled = scale_y_data(y_train, y_val, y_test,
    #                        y_err_train=y_err_train, y_err_val=y_err_val, y_err_test=y_err_test,
    #                        return_scaler=True)
    # y_train_scaled, y_val_scaled, y_test_scaled, \
    #            y_err_train_scaled, y_err_val_scaled, y_err_test_scaled, scaler = ys_scaled
    
    ### Run inference
    if run_moment:
        #tag_inf = f'{tag_data}_ntrain{n_train}_scalecovminmax'
        #tag_inf = f'{tag_data}_ntrain{n_train}_eigs'
        tag_run = ''
        
        run_mode_mean = 'best'
        sweep_name_mean = 'rand10'
        #run_mode_mean = 'sweep'
        #sweep_name_mean = 'biaszenp4rand10'
        # run_mode_mean = 'load'
        # tag_run += '_best-rand10'
        # sweep_name_mean = None
        
        #run_mode_cov = 'single'
        #sweep_name_cov = None
        #run_mode_cov = 'sweep'
        run_mode_cov = 'best'
        sweep_name_cov = 'rand10'
        # run_mode_cov = 'load'
        # tag_run += '_bestcov-rand10'
        # sweep_name_cov = None
        
        if run_mode_mean == 'sweep':
            tag_run += f'_sweep-{sweep_name_mean}'
        elif run_mode_mean == 'best':
            tag_run += f'_best-{sweep_name_mean}'

        if run_mode_cov == 'sweep':
            tag_run += f'_sweepcov-{sweep_name_cov}'
        elif run_mode_cov == 'best':
            tag_run += f'_bestcov-{sweep_name_cov}'
            
        tag_inf = f'{tag_data}_ntrain{n_train}_direct{tag_run}'
        print("tag_inf", tag_inf)
        # run_mode_mean = 'single'
        # sweep_name_mean = None
        # run_mode_cov = None
        # sweep_name_cov = None
        # tag_inf = f'{tag_data}_ntrain{n_train}_direct'
        
        moment_network = mn.MomentNetwork(theta_train=theta_train, y_train_unscaled=y_train, y_err_train_unscaled=y_err_train,
                    theta_val=theta_val, y_val_unscaled=y_val, y_err_val_unscaled=y_err_val,
                    theta_test=theta_test, y_test_unscaled=y_test, y_err_test_unscaled=y_err_test,
                    tag_mn=tag_inf,
                    run_mode_mean=run_mode_mean, run_mode_cov=run_mode_cov,
                    sweep_name_mean=sweep_name_mean, sweep_name_cov=sweep_name_cov)
        # moment_network = mn.MomentNetwork(theta_train=theta_train, y_train=y_train_scaled, y_err_train=y_err_train_scaled,
        #                     theta_val=theta_val, y_val=y_val_scaled, y_err_val=y_err_val_scaled,
        #                     theta_test=theta_test, y_test=y_test_scaled, y_err_test=y_err_test_scaled,
        #                     tag_mn=tag_inf,
        #                     run_mode_mean=run_mode_mean, run_mode_cov=run_mode_cov,
        #                     sweep_name_mean=sweep_name_mean, sweep_name_cov=sweep_name_cov)
        #moment_network.run(max_epochs_mean=5, max_epochs_cov=5)
        moment_network.run(max_epochs_mean=2000, max_epochs_cov=2000)
        moment_network.evaluate_test_set()
        print(f"Saved results to {moment_network.dir_mn}")
    

    if run_emcee:
        # TODO deal with how bias_params are handled now that they might be a vector
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
        
        
def test_pk_inference():
    run_moment = True
    run_sbi = False
    run_emcee = False
    run_dynesty = False
    
    # for likelihood methods only
    idxs_obs = np.arange(1)

    ### Load data
    data_mode = 'muchisimocksPk'

    tag_mocks_train = '_p5_n10000'
    #tag_pk_train = '_b1000'
    tag_pk_train = '_biaszen_p4_n10000'

    tag_mocks = '_fixedcosmo'
    #tag_pk = '_b1000'
    #mode_bias_vector = 'single'
    tag_pk = '_biaszen_p4_n1000'
    mode_bias_vector = 'LH'
    tag_datagen = f'{tag_mocks_train}{tag_pk_train}'
    
    param_dict, y, gaussian_error_pk, k, bias_vector = load_data_muchisimocksPk_fixedcosmo(tag_mocks,
                                            tag_pk,
                                            mode_bias_vector=mode_bias_vector
                                            )

    tag_data = '_'+data_mode + tag_datagen
    print(y.shape)


    ### Run inference
    if run_moment:
        #tag_inf = f'{tag_data}_ntrain{n_train}_scalecovminmax'
        #tag_inf = f'{tag_data}_ntrain{n_train}_eigs'
        tag_run = ''
        
        run_mode_mean = 'load'
        tag_run += '_best-rand10'
        sweep_name_mean = None
        
        run_mode_cov = 'load'
        tag_run += '_bestcov-rand10'
        sweep_name_cov = None

            
        n_train = 8000
        tag_inf = f'{tag_data}_ntrain{n_train}_direct{tag_run}'
        print("tag_inf", tag_inf)
        # run_mode_mean = 'single'
        # sweep_name_mean = None
        # run_mode_cov = None
        # sweep_name_cov = None
        # tag_inf = f'{tag_data}_ntrain{n_train}_direct'
        
        moment_network = mn.MomentNetwork(
                            tag_mn=tag_inf,
                            run_mode_mean=run_mode_mean, run_mode_cov=run_mode_cov,
                            sweep_name_mean=sweep_name_mean, sweep_name_cov=sweep_name_cov)
        moment_network.run() #need this to do the loading
        moment_network.evaluate_test_set(y_test_unscaled=y, tag_test=tag_mocks)
        
        y_mean = np.mean(y, axis=0)
        moment_network.evaluate_test_set(y_test_unscaled=y_mean, tag_test=f'{tag_mocks}_mean')
        print(f"Saved results to {moment_network.dir_mn}")
    
            

def load_data_muchisimocksPk(tag_mocks, tag_pk, mode_bias_vector='single'):       
    
    if 'fixedcosmo' in tag_mocks:
        param_dict = utils.cosmo_dict_quijote
    else:
        dir_params = '../data/params'
        fn_params = f'{dir_params}/params_lh{tag_mocks}.txt'
        params_df = pd.read_csv(fn_params, index_col=0)
    
    # NOTE for now theta is only cosmo params! 
    # may want to add bias params too 
    
    param_names = params_df.columns.tolist()
    #idxs_LH = params_df.index.tolist()
    dir_pks = f'/scratch/kstoreyf/muchisimocks/data/pks_mlib/pks{tag_mocks}{tag_pk}'
    idxs_LH = np.array([idx_LH for idx_LH in params_df.index.values 
                        if os.path.exists(f"{dir_pks}/pk_{idx_LH}.npy")])
    assert len(idxs_LH) > 0, f"No pks found in {dir_pks}!"
    
    if mode_bias_vector == 'single':
        fn_bias_vector = f'{dir_pks}/bias_params.txt'
        bias_vector = np.loadtxt(fn_bias_vector)
    else:
        # TODO update as needed
        bias_vector = None

    fn_rands = f'{dir_params}/randints{tag_mocks}.npy'
    random_ints = np.load(fn_rands)
    random_ints = random_ints[idxs_LH]

    theta, Pk, gaussian_error_pk = [], [], []
    for idx_LH in idxs_LH:
        fn_pk = f'{dir_pks}/pk_{idx_LH}.npy'
        pk_obj = np.load(fn_pk, allow_pickle=True).item()
        Pk.append(pk_obj['pk'])
        gaussian_error_pk.append(pk_obj['pk_gaussian_error'])
        if 'fixedcosmo' in tag_mocks:
            theta.append(param_dict.values)
        else:
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


def load_data_muchisimocksPk_fixedcosmo(tag_mocks, tag_pk, mode_bias_vector='single'):       
   
    idxs_mock = np.array([idx_mock for idx_mock in range(1000) if os.path.exists(f"/scratch/kstoreyf/muchisimocks/data/pks_mlib/pks{tag_mocks}{tag_pk}/pk_{idx_mock}.npy")])
    
    dir_pks = f'/scratch/kstoreyf/muchisimocks/data/pks_mlib/pks{tag_mocks}{tag_pk}'
    if mode_bias_vector == 'single':
        fn_bias_vector = f'{dir_pks}/bias_params.txt'
        bias_vector = np.loadtxt(fn_bias_vector)
    else:
        # TODO update as needed
        bias_vector = None
        
    param_dict = utils.cosmo_dict_quijote


    Pk, gaussian_error_pk = [], []
    for idx_mock in idxs_mock:
        fn_pk = f'{dir_pks}/pk_{idx_mock}.npy'
        pk_obj = np.load(fn_pk, allow_pickle=True).item()
        Pk.append(pk_obj['pk'])
        gaussian_error_pk.append(pk_obj['pk_gaussian_error'])

    Pk = np.array(Pk)
    gaussian_error_pk = np.array(gaussian_error_pk)
    k = pk_obj['k'] # all ks should be same so just grab one

    # TODO do i want / need this here?
    mask = np.all(Pk>0, axis=0)
    print(f"{np.sum(np.any(Pk<=0, axis=1))}/{Pk.shape[0]} have at least one non-positive Pk value")
    print(f"Masking columns {np.where(~mask)[0]}")
    Pk = Pk[:,mask]
    gaussian_error_pk = gaussian_error_pk[:,mask]
    k = k[mask]
    return param_dict, Pk, gaussian_error_pk, k, bias_vector

        
def load_data_muchisimocks3D(tag_mocks):       
     
    print("Loading 3D data")
    dir_params = '../data/params'
    fn_params = f'{dir_params}/params_lh{tag_mocks}.txt'
    params_df = pd.read_csv(fn_params, index_col=0)
    param_names = params_df.columns.tolist()
    idxs_LH = params_df.index.tolist()

    dir_mocks = f'/scratch/kstoreyf/muchisimocks/muchisimocks_lib{tag_mocks}'
    tag_fields = '_deconvolved'
    tag_fields_extra = ''
    n_grid_orig = 512
    # TODO deal with bias vector better!
    bias_vector = np.array([1.,0.,0.,0.])

    fn_rands = f'{dir_params}/randints{tag_mocks}.npy'
    random_ints = np.load(fn_rands)

    theta, fields, errors = [], [], []
    for idx_LH in idxs_LH:
        fn_fields = f'{dir_mocks}/LH{idx_LH}/bias_fields_eul{tag_fields}_{idx_LH}{tag_fields_extra}.npy'
        bias_terms_eul = np.load(fn_fields)
        tracer_field = utils.get_tracer_field(bias_terms_eul, bias_vector, n_grid_norm=n_grid_orig)
        fields.append(tracer_field)
        # TODO what is my error??
        param_vals = params_df.loc[idx_LH].values
        theta.append(param_vals)

    fields = np.array(fields)
    theta = np.array(theta)
    errors = np.ones_like(fields) # not ideal but just for now
    
    # None is where k is for Pk
    return theta, fields, errors, None, param_names, bias_vector, random_ints


        
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