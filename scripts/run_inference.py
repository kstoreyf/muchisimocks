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

import generate_emuPks as genP


def main():
    #run_pk_inference()
    test_pk_inference()
    #run_likelihood_inference()



def train_likefree_inference():
    run_moment = False
    run_sbi = True
    # train test split, and number of rlzs per cosmo
    n_train = None

    ### Load data
    #data_mode = 'muchisimocksPk'
    data_mode = 'emuPk'
    tag_mocks = '_p5_n10000' #for emu, formerly tag_emuPk
    tag_data_extra = '_boxsize500' #formerly tag_errG for data
    kwargs = {'n_rlzs_per_cosmo': n_rlzs_per_cosmo}

    ### TODO clean all this up, now can just load train vs test and then do val split later?? 
    assert data_mode in ['emuPk', 'muchisimocksPk']
    
    # if data_mode == 'emuPk':
    #     #n_rlzs_per_cosmo = 9
    #     #tag_emuPk = '_5param'
    #     tag_emuPk = '_2param'
    #     tag_errG = f'_boxsize500'
    #     tag_datagen = f'{tag_emuPk}{tag_errG}_nrlzs{n_rlzs_per_cosmo}'
    #     n_rlzs_per_cosmo = 1
    #     kwargs = {'n_rlzs_per_cosmo': n_rlzs_per_cosmo}
        
    #     theta, y, y_err, k, param_names, bias_params, random_ints = load_data_emuPk(tag_emuPk, 
    #                                                     tag_errG, tag_datagen,
    #                                                     n_rlzs_per_cosmo=n_rlzs_per_cosmo)
    # elif data_mode == 'muchisimocksPk':
    #     tag_mocks = '_p5_n10000'
    #     tag_pk = '_b1000'
    #     mode_bias_vector = 'single'
    #     #tag_pk = '_biaszen_p4_n10000'
    #     #mode_bias_vector = 'LH'
    #     tag_datagen = f'{tag_mocks}{tag_pk}'
    #     theta, y, y_err, k, param_names, bias_params, random_ints = load_data_muchisimocksPk(tag_mocks,
    #                                                                                          tag_pk,
    #                                                                                          mode_bias_vector=mode_bias_vector
    #                                                                                          )
    tag_datagen = f'{tag_mocks}{tag_data_extra}'
    theta, y, y_err, k, param_names, bias_params, random_ints = \
                data_loader.load_data(data_mode, tag_params, tag_biasparams, tag_datagen, #mode_bias_vector=mode_bias_vector,
                                      kwargs=kwargs_data)
    
    tag_data = '_'+data_mode + tag_datagen

    # split into train and validation - but for SBI, these just get lumped back together
    idxs_train, idxs_val, _ = utils.idxs_train_val_test(random_ints, frac_train=0.9, frac_val=0.1, frac_test=0.0,
                        N_tot=len(theta))
    # fixing absolute size of validation set, as per chatgpt's advice: https://chatgpt.com/share/678e9ef1-a574-8002-adcb-4929630fbf01
    if n_train is not None:
        idxs_train = idxs_train[n_train]
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
        # moment_network = mn.MomentNetwork(theta_train=theta_train, y_train=y_train_scaled, y_err_train=y_err_train_scaled,
        #                     theta_val=theta_val, y_val=y_val_scaled, y_err_val=y_err_val_scaled,
        #                     theta_test=theta_test, y_test=y_test_scaled, y_err_test=y_err_test_scaled,
        #                     tag_mn=tag_inf,
        #                     run_mode_mean=run_mode_mean, run_mode_cov=run_mode_cov,
        #                     sweep_name_mean=sweep_name_mean, sweep_name_cov=sweep_name_cov)
        #moment_network.run(max_epochs_mean=5, max_epochs_cov=5)
        moment_network.run(max_epochs_mean=2000, max_epochs_cov=2000)
        print(f"Saved results to {moment_network.dir_mn}")
    
    if run_sbi:
        run_mode = 'single'
        tag_inf = f'{tag_data}_ntrain{n_train}'
        sbi_network = sbi_model.SBIModel(theta_train=theta_train, y_train_unscaled=y_train, y_err_train_unscaled=y_err_train,
                    theta_val=theta_val, y_val_unscaled=y_val, y_err_val_unscaled=y_err_val,
                    theta_test=theta_test, y_test_unscaled=y_test, y_err_test_unscaled=y_err_test,
                    tag_sbi=tag_inf, run_mode=run_mode,
                    param_names=param_names)
        sbi_network.run()


    if run_emcee or run_dynesty:
        # for likelihood methods, set up scaler here (TODO maybe better to do in MCMC?)
        # for SBI, doing in classes, bc need to make sure to save the scaler with the model
        ### Scale data
        ys_scaled = scale_y_data(y_train, y_val, y_test,
                               y_err_train=y_err_train, y_err_val=y_err_val, y_err_test=y_err_test,
                               return_scaler=True)
        y_train_scaled, y_val_scaled, y_test_scaled, \
                   y_err_train_scaled, y_err_val_scaled, y_err_test_scaled, scaler = ys_scaled


            

        
        
def test_pk_inference():
    run_moment = False
    run_sbi = True
    run_emcee = False
    run_dynesty = False
    
    # for likelihood methods only
    idxs_obs = np.arange(1)

    ### Load data
    #data_mode = 'muchisimocksPk'
    data_mode = 'emuPk'
    assert data_mode in ['emuPk', 'muchisimocksPk']
    
    tag_params = '_test_p5_n1000'
    tag_biasparams = '_b1000_p0_n1'
    
    theta_cosmo, theta_bias, cosmo_params_fixed, bias_params_fixed, \ 
        cosmo_param_names, bias_param_names, y, y_err, k, random_ints = \
        data_loader.load_data(data_mode, tag_params, tag_biasparams, tag_datagen, #mode_bias_vector=mode_bias_vector,
                            kwargs=kwargs_data)
                
    if data_mode == 'emuPk':
        #n_rlzs_per_cosmo = 9
        n_rlzs_per_cosmo = 1
        #tag_emuPk = '_5param'
        #tag_emuPk = '_2param'
        tag_emuPk = '_fixedcosmo_n1000'
        tag_errG = f'_boxsize500'

        test_noiseless = False
        tag_datagen = f'{tag_emuPk}{tag_errG}_nrlzs{n_rlzs_per_cosmo}'
        
        theta, Pk, gaussian_error_pk, k, param_names_all, bias_params, random_ints, \
               theta_noiseless, Pk_noiseless, gaussian_error_pk_noiseless = load_data_emuPk(tag_emuPk, 
                                                        tag_errG, tag_datagen,
                                                        n_rlzs_per_cosmo=n_rlzs_per_cosmo,
                                                        return_noiseless=True)
        
        tag_test = tag_emuPk
        # for fixedcosmo, we decide which parameters to sample over
        if 'fixedcosmo' in tag_emuPk:
            param_names = ['omega_cold', 'sigma8_cold']
            #param_names = ['omega_cold', 'sigma8_cold', 'hubble', 'ns', 'omega_baryon']
            tag_test += f'_{len(param_names)}param'
            theta = np.array([theta[:,param_names_all.index(pn)] for pn in param_names]).T
            
        
        if test_noiseless:
            theta_test = theta[idxs_obs]
            y_test = Pk_noiseless[idxs_obs]
            y_err_test = gaussian_error_pk_noiseless[idxs_obs]
            tag_test += '_noiseless'
        else:
            theta_test = theta
            y_test = Pk
            y_err_test = gaussian_error_pk


    elif data_mode == 'muchisimocksPk':
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
        
    if run_sbi:
        #n_train = 8000
        #tag_inf = f'{tag_data}_ntrain{n_train}'
        tag_inf = '_emuPk_2param_boxsize500_nrlzs1_ntrain8000'
        sbi_network = sbi_model.SBIModel(
                    theta_test=theta_test, y_test_unscaled=y_test, y_err_test_unscaled=y_err_test,
                    tag_sbi=tag_inf, run_mode='load',
                    param_names=param_names       
                    )
        sbi_network.run() #need this to do the loading
        # TODO make this work for both emu and muchisimocks
        #sbi_network.evaluate_test_set(y_test_unscaled=y, tag_test=tag_emupk)
        
        sbi_network.evaluate_test_set(y_test_unscaled=y_test, tag_test=tag_test)
        
        
    
    
    

def run_likelihood_inference():

    run_emcee = False
    run_dynesty = True
    
    # for likelihood methods only
    idxs_obs = np.arange(1)

    ### Load data
    #data_mode = 'muchisimocksPk'
    data_mode = 'emuPk'
    assert data_mode in ['emuPk', 'muchisimocksPk']
    
    tag_params = '_test_p2_n1000'
    tag_biasparams = '_b1000_p0_n1'
    kwargs_data = {}
    
    theta_cosmo, theta_bias, cosmo_params_fixed, bias_params_fixed, \ 
        cosmo_param_names, bias_param_names, y, y_err, k, random_ints = \
        data_loader.load_data(data_mode, tag_params, tag_biasparams, tag_datagen, #mode_bias_vector=mode_bias_vector,
                            kwargs=kwargs_data)
    
    tag_test = ''
    if data_mode == 'emuPk':
        #n_rlzs_per_cosmo = 9
        n_rlzs_per_cosmo = 1
        #tag_emuPk = '_5param'
        #tag_emuPk = '_2param'
        tag_emuPk = '_fixedcosmo_n1000'
        tag_errG = f'_boxsize500'
        
        test_noiseless = True
        tag_datagen = f'{tag_emuPk}{tag_errG}_nrlzs{n_rlzs_per_cosmo}'
        
        theta, Pk, gaussian_error_pk, k, param_names, bias_params, random_ints, \
               theta_noiseless, Pk_noiseless, gaussian_error_pk_noiseless = load_data_emuPk(tag_emuPk, 
                                                        tag_errG, tag_datagen,
                                                        n_rlzs_per_cosmo=n_rlzs_per_cosmo,
                                                        return_noiseless=True)
        
        # for fixedcosmo, we decide which parameters to sample over
        if 'fixedcosmo' in tag_emuPk:
            param_names = ['omega_cold', 'sigma8_cold']
            #param_names = ['omega_cold', 'sigma8_cold', 'hubble', 'ns', 'omega_baryon']
            tag_test += f'_{len(param_names)}param'

        if test_noiseless:
            y_test = Pk_noiseless[idxs_obs]
            y_err_test = gaussian_error_pk_noiseless[idxs_obs]
            tag_test += '_noiseless'
        else:
            y_test = Pk[idxs_obs]
            y_err_test = gaussian_error_pk[idxs_obs]


    # not using training data for likelihood methods, so unclear how to scale; 
    # let's just do log for now
    # (maybe really should take reasonable bounds from training data?)
    
    scaler = scl.Scaler('log')
    scaler.fit(y_test)
    y_test_scaled = scaler.scale(y_test)
    y_err_test_scaled = scaler.scale_error(y_err_test, y_test)

    tag_data = '_'+data_mode + tag_datagen + tag_test
           
           
    dir_emus_lbias = '/home/kstoreyf/external' #hyperion path
    emu, emu_bounds, emu_param_names_all = utils.load_emu(dir_emus_lbias=dir_emus_lbias)    
        
    if run_dynesty:
        
        import mcmc
    
        #dict_bounds = {name: emu_bounds[emu_param_names_all.index(name)] for name in param_names}
        # for emulator, the emu param names are the param names, but not so for muchisimocks necessarily
        # not handled bc havent tried likelihood for muchisimocks yet?
        emu_param_names = param_names
        cosmo_params = utils.setup_cosmo_emu()
        
        tag_inf = f'{tag_data}'
      
        for idx_obs in idxs_obs:
            
            y_data_unscaled = y_test[idx_obs]
            y_data = y_test_scaled[idx_obs]

            # err_1p = 0.01*y_data_unscaled
            # err_1p_scaled = scaler.scale_error(err_1p, y_data_unscaled)
            # err_gaussian_scaled = y_err_test_scaled[idx_obs]
            # var = err_gaussian_scaled**2 + err_1p_scaled**2
            # cov_inv = np.diag(1/var)
            cov_inv = np.diag(np.ones(len(y_data)))
            tag_inf += f'_covnone'
        
            mcmc.evaluate_mcmc(idx_obs, y_data, cov_inv, scaler, 
                   emu, k, 
                   cosmo_params_fixed, bias_params_fixed, 
                   cosmo_param_names_vary, bias_param_names_vary,
                   dict_bounds_cosmo, dict_bounds_bias,
                   tag_inf='', n_threads=8, framework='dynesty')
           
    # UNTESTED after refactor
    if run_emcee:
        # TODO deal with how bias_params are handled now that they might be a vector
        import mcmc
        dir_emus_lbias = '/home/kstoreyf/external' #hyperion path
        emu, emu_bounds, emu_param_names = utils.load_emu(dir_emus_lbias=dir_emus_lbias)
        dict_bounds = {name: emu_bounds[emu_param_names.index(name)] for name in param_names}
        cosmo_params = utils.setup_cosmo_emu()   
            
        tag_inf = f'{tag_data}'
        
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