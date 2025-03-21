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
import sbi_model
import scaler_custom as scl
import data_loader

import generate_params_lh as gplh


def main():
    train_likefree_inference()
    test_likefree_inference()
    #run_likelihood_inference()



def train_likefree_inference():
    run_moment = False
    run_sbi = True

    ### Set up data
    data_mode = 'emuPk'
    #data_mode = 'muchisimocksPk'
    n_train = 10000 #if None, uses all
    tag_params = '_p5_n10000' #for emu, formerly tag_emuPk
    #tag_biasparams = '_b1000_p0_n1'
    tag_biasparams = '_biaszen_p4_n10000'
    n_rlzs_per_cosmo = 1
    
    if data_mode == 'emuPk':
        tag_errG = '_boxsize1000'
        tag_datagen = f'{tag_errG}_nrlzs{n_rlzs_per_cosmo}'
        kwargs_data = {'n_rlzs_per_cosmo': n_rlzs_per_cosmo,
                    'tag_errG': tag_errG,
                    'tag_datagen': tag_datagen}
    elif data_mode == 'muchisimocksPk':
        tag_datagen = ''
        kwargs_data = {'tag_datagen': tag_datagen}

    # no tag_noiseless here for now, bc not training on noiseless data
    tag_data = '_'+data_mode + tag_params + tag_biasparams + tag_datagen

    ### Load data and parameters
    # don't need the fixed params for training!
    k, y, y_err, params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, random_ints, random_ints_bias = \
                data_loader.load_data(data_mode, tag_params, tag_biasparams,
                                      kwargs=kwargs_data)
    mask = data_loader.get_Pk_mask(tag_data, Pk=y)
    print(mask)
    print(f"Masked {np.sum(mask)} out of {len(mask)} bins")

    k, y, y_err = k[mask], y[:,mask], y_err[:,mask]
    print(k.shape, y.shape, y_err.shape)

    # get bounds dict
    _, dict_bounds_cosmo, _ = gplh.define_LH_cosmo(tag_params)
    _, dict_bounds_bias, _ = gplh.define_LH_bias(tag_biasparams)
    dict_bounds = {**dict_bounds_cosmo, **dict_bounds_bias}

    theta, param_names = data_loader.param_dfs_to_theta(params_df, biasparams_df, n_rlzs_per_cosmo=n_rlzs_per_cosmo)
    print(param_names)

    # split into train and validation - but for SBI, these just get lumped back together
    # TODO deal properly with random ints - may break
    idxs_train, idxs_val, _ = utils.idxs_train_val_test(random_ints, frac_train=0.9, frac_val=0.1, frac_test=0.0,
                        N_tot=len(theta))
    # fixing absolute size of validation set, as per chatgpt's advice: https://chatgpt.com/share/678e9ef1-a574-8002-adcb-4929630fbf01
    if n_train is None:
        n_train = len(idxs_train) # just get what it is for the full set
    else:
        idxs_train = idxs_train[:n_train]
    theta_train, theta_val = theta[idxs_train], theta[idxs_val]
    y_train, y_val = y[idxs_train], y[idxs_val]
    y_err_train, y_err_val = y_err[idxs_train], y_err[idxs_val]
        
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
                    tag_mn=tag_inf,
                    run_mode_mean=run_mode_mean, run_mode_cov=run_mode_cov,
                    sweep_name_mean=sweep_name_mean, sweep_name_cov=sweep_name_cov)
        #moment_network.run(max_epochs_mean=5, max_epochs_cov=5)
        moment_network.run(max_epochs_mean=2000, max_epochs_cov=2000)
        print(f"Saved results to {moment_network.dir_mn}")
    
    if run_sbi:
        run_mode = 'single'
        tag_inf = f'{tag_data}_ntrain{n_train}'
        sbi_network = sbi_model.SBIModel(theta_train=theta_train, y_train_unscaled=y_train, y_err_train_unscaled=y_err_train,
                    theta_val=theta_val, y_val_unscaled=y_val, y_err_val_unscaled=y_err_val,
                    tag_sbi=tag_inf, run_mode=run_mode,
                    param_names=param_names, dict_bounds=dict_bounds)
        sbi_network.run()


        
        
def test_likefree_inference():
    run_moment = False
    run_sbi = True

    #idxs_obs = np.arange(1)
    idxs_obs = None
    evaluate_mean = True # this will be in additon to the idxs_obs!
    #data_mode = 'muchisimocksPk'

    ### Select trained model
    data_mode = 'emuPk'
    #data_mode = 'muchisimocksPk'
    
    # train params
    tag_params = '_p5_n10000'
    #tag_biasparams = '_b1000_p0_n1'
    tag_biasparams = '_biaszen_p4_n10000'
    n_rlzs_per_cosmo = 1
    n_train = 10000
    
    # test params
    tag_params_test = '_quijote_p0_n1000'
    tag_biasparams_test = '_b1000_p0_n1'
    #tag_biasparams_test = '_biaszen_p4_n1000'
        
    # this if-else is just so it's easier for me to switch between the two; may not need
    if data_mode == 'emuPk':
        # train
        tag_errG = '_boxsize1000'
        tag_datagen = f'{tag_errG}_nrlzs{n_rlzs_per_cosmo}'
        # test
        tag_errG = '_boxsize1000'
        tag_noiseless = ''
        #tag_noiseless = '_noiseless' # if use noiseless, set evaluate_mean=False (?)
        tag_datagen_test = f'{tag_errG}_nrlzs{n_rlzs_per_cosmo}'
        kwargs_data_test = {'n_rlzs_per_cosmo': n_rlzs_per_cosmo,
                            'tag_errG': tag_errG,
                            'tag_datagen': tag_datagen,
                            'tag_noiseless': tag_noiseless}
    elif data_mode == 'muchisimocksPk':
        # train
        tag_datagen = ''
        # test
        tag_noiseless = ''
        tag_datagen_test = ''
        kwargs_data_test = {'tag_datagen': tag_datagen}
    
    # don't need train kwargs here bc not actually loading the data; just getting tag to reload model
    tag_data_train = '_'+data_mode + tag_params + tag_biasparams + tag_datagen
    tag_data_test = '_'+data_mode + tag_params_test + tag_biasparams_test + tag_datagen_test + tag_noiseless
    mask = data_loader.get_Pk_mask(tag_data_train)

    ### Load data and parameters
    # our setup is such that that the test set is a separate dataset, so no need to split
    # don't need theta either - just predicting, not comparing
    k, y, y_err, params_df, cosmo_param_dict_fixed, biasparams_df, bias_param_dict_fixed, random_ints, random_ints_bias = \
                data_loader.load_data(data_mode, tag_params_test, tag_biasparams_test,
                                      kwargs=kwargs_data_test)
    k, y, y_err = k[mask], y[:,mask], y_err[:,mask]

    param_names_train = data_loader.get_param_names(tag_params=tag_params, tag_biasparams=tag_biasparams)

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

        tag_inf = f'{tag_data_train}_ntrain{n_train}_direct{tag_run}'
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
        moment_network.evaluate_test_set(y_test_unscaled=y, tag_test=tag_data_test)
        
        if evaluate_mean:
            y_mean = np.mean(y, axis=0)
            moment_network.evaluate_test_set(y_test_unscaled=y_mean, tag_test=f'{tag_data_test}_mean')
        
    if run_sbi:
        #n_train = 8000
        tag_inf = f'{tag_data_train}_ntrain{n_train}'
        #tag_inf = f'{tag_data_train}_ntrain{n_train}_nsf'
        #tag_inf = '_emuPk_2param_boxsize500_nrlzs1_ntrain8000'
        sbi_network = sbi_model.SBIModel(
                    tag_sbi=tag_inf, run_mode='load',
                    param_names=param_names_train,
                    )
        sbi_network.run() #need this to do the loading
        # TODO make this work for both emu and muchisimocks # ?? not sure what this means rn
        if idxs_obs is None:
            y_obs = y
        else:
            y_obs = y[idxs_obs]
            
        # maybe should load this in as a separate dataset, but for now seems fine to do this way
        if evaluate_mean:
            y_mean = np.mean(y, axis=0)        
            sbi_network.evaluate_test_set(y_test_unscaled=np.atleast_2d(y_mean), tag_test=f'{tag_data_test}_mean')
        
        # run on full test set
        sbi_network.evaluate_test_set(y_test_unscaled=y_obs, tag_test=tag_data_test)


def run_likelihood_inference():

    mcmc_framework = 'dynesty'
    #mcmc_framework = 'emcee'
    
    # for likelihood methods only
    idxs_obs = np.arange(1)
    evaluate_mean = False # if true, idx_obs ignored and mean of whole test dataset taken

    ### Set up data
    #data_mode = 'muchisimocksPk'
    data_mode = 'emuPk'
    #tag_params = '_p2_n10000' 
    tag_params = '_quijote_p0_n1000'
    tag_biasparams = '_b1000_p0_n1'
    n_rlzs_per_cosmo = 1
    tag_errG = '_boxsize1000'
    #tag_noiseless = ''
    tag_noiseless = '_noiseless' # for emulator, probs want to use noiseless rather than evaluate mean
    tag_datagen = f'{tag_errG}_nrlzs{n_rlzs_per_cosmo}'
    tag_data = '_'+data_mode + tag_params + tag_biasparams + tag_datagen + tag_noiseless

    # for likelihood methods, we decide which parameters to sample over! 
    # (will have to change when have a test set not generated with these params, e.g. hydro)
    #cosmo_param_names_vary = ['omega_cold', 'sigma8_cold']
    cosmo_param_names_vary = ['omega_cold', 'sigma8_cold', 'hubble', 'ns', 'omega_baryon']
    bias_param_names_vary = []
    tag_inf = f'{tag_data}_mcmctest_p{len(cosmo_param_names_vary)}_b{len(bias_param_names_vary)}'
            
    kwargs_data = {'n_rlzs_per_cosmo': n_rlzs_per_cosmo,
                   'tag_errG': tag_errG,
                   'tag_datagen': tag_datagen,
                   'tag_noiseless': tag_noiseless}
    if evaluate_mean:
        assert 'p0' in tag_params, "If you're evaluating the mean, don't you want fixed cosmo?"
    
    ### Load data and parameters
    # theta nor random ints needed for likelihood inf
    k, y, y_err, params_df, cosmo_param_dict_fixed, biasparams_df, bias_param_dict_fixed, random_ints, random_ints_bias = \
                data_loader.load_data(data_mode, tag_params, tag_biasparams,
                                      kwargs=kwargs_data)
                
    ### get covariance
    tag_params_cov = '_p2_n10000'
    tag_biasparams_cov = '_b1000_p0_n1'
    kwargs_data_cov = {'n_rlzs_per_cosmo': n_rlzs_per_cosmo,
                   'tag_errG': tag_errG,
                   'tag_datagen': tag_datagen,
                   'tag_noiseless': ''} #don't want noiseless here bc want to match SBI training
    k_cov, y_cov, y_err_cov, params_df_cov, cosmo_param_dict_fixed_cov, biasparams_df_cov, bias_param_dict_fixed_cov, random_ints_cov, random_ints_bias_cov = \
                data_loader.load_data(data_mode, tag_params_cov, tag_biasparams_cov,
                                      kwargs=kwargs_data_cov)
    cov = np.cov(y_cov.T)

    
    
    if evaluate_mean:
        print("Evaluating mean of test set! idxs_obs ignored")
        ys_obs = np.array([np.mean(y, axis=0)])
        ys_err_obs = np.array([np.mean(y_err, axis=0)]) # is it fine to take the mean of the err?
    else:
        # just grab the ones we're going to loop over to observe
        ys_obs = y[idxs_obs]
        ys_err_obs = y_err[idxs_obs]
        
    # get bounds
    _, dict_bounds_cosmo, _ = gplh.define_LH_cosmo(tag_params)
    _, dict_bounds_bias, _ = gplh.define_LH_bias(tag_biasparams)
    print("bounds")
    print(dict_bounds_cosmo)
    print(dict_bounds_bias)

    # not using training data for likelihood methods, so unclear how to scale; 
    # let's just do log for now
    # (maybe really should take reasonable bounds from training data?)
    scaler = scl.Scaler('log')
    scaler.fit(ys_obs)
    ys_obs_scaled = scaler.scale(ys_obs)
    ys_err_obs_scaled = scaler.scale_error(ys_err_obs, ys_obs)
           
    dir_emus_lbias = '/home/kstoreyf/external' #hyperion path
    emu, emu_bounds, emu_param_names_all = utils.load_emu(dir_emus_lbias=dir_emus_lbias)    
    
    for i in range(len(ys_obs)):
    
        if evaluate_mean:
            idx_obs = 0 #shouldn't matter because should just be using fixed set! 
            tag_obs = '_mean'
        else:
            idx_obs = idxs_obs[i]
            tag_obs = None # will be automatic
                
        # get single observation
        y_obs_unscaled = ys_obs[i]
        y_obs = ys_obs_scaled[i]
        
        # gather the fixed parameters (may be from the varied df bc for likelihood approach
        # we can choose which params to fix and sample, indep of test parameters)
        cosmo_param_dict_fixed_obs = cosmo_param_dict_fixed.copy()
        if params_df is not None:
            cosmo_param_dict_fixed_obs.update(params_df.loc[idx_obs].to_dict())
        for pn in cosmo_param_names_vary:
            cosmo_param_dict_fixed.pop(pn)
        
        bias_param_dict_fixed_obs = bias_param_dict_fixed.copy()
        if biasparams_df is not None:
            bias_param_dict_fixed_obs.update(biasparams_df.loc[idx_obs].to_dict())
        for pn in bias_param_names_vary:
            bias_param_dict_fixed_obs.pop(pn)  

        # construct covariance matrix
        err_1p = 0.01*y_obs_unscaled
        err_1p_scaled = scaler.scale_error(err_1p, y_obs_unscaled)
        err_gaussian_scaled = ys_err_obs_scaled[i]
        var = err_gaussian_scaled**2 + err_1p_scaled**2
        cov_inv = np.diag(1/var)
        #cov_inv = np.diag(np.ones(len(y_data)))
        #tag_inf += f'_covnone'
    
        import mcmc
        mcmc.evaluate_mcmc(idx_obs, y_obs, cov_inv, scaler, 
                emu, k, 
                cosmo_param_dict_fixed, bias_param_dict_fixed, 
                cosmo_param_names_vary, bias_param_names_vary,
                dict_bounds_cosmo, dict_bounds_bias,
                tag_inf=tag_inf, tag_obs=tag_obs, 
                n_threads=8, mcmc_framework=mcmc_framework)
           

         
def scale_y_data(y_train, y_val, y_test,
                 y_err_train=None, y_err_val=None, y_err_test=None,
                 return_scaler=True):
    scaler = scl.Scaler('log_minmax')
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