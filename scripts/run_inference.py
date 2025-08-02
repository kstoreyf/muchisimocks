import os
os.environ["OMP_NUM_THREADS"] = str(1)

import numpy as np

from multiprocessing import Pool, cpu_count

import argparse
import pandas as pd
import yaml

import sys
sys.path.append('/dipc/kstoreyf/muchisimocks/scripts')
import utils
import moment_network as mn
import moment_network_dataloader as mndl
import sbi_model
import scaler_custom as scl
import data_loader
import generate_params as genp


def main():
    
    parser = argparse.ArgumentParser(description="Run inference with config files.")
    parser.add_argument("-tr", "--config-train", type=str, help="Path to the training YAML configuration file.")
    parser.add_argument("-te", "--config-test", type=str, help="Path to the testing YAML configuration file.")
    parser.add_argument("-l", "--config-runlike", type=str, help="Path to the runlike YAML configuration file.")
    args = parser.parse_args()

    # Run training if a training config file is provided
    if args.config_train:
        with open(args.config_train, "r") as file:
            train_config = yaml.safe_load(file)
        train_likefree_inference(train_config)

    # Run testing if a testing config file is provided
    if args.config_test:
        with open(args.config_test, "r") as file:
            test_config = yaml.safe_load(file)
        test_likefree_inference(test_config)

    # WARNING not implemented yet !
    if args.config_runlike:
        print("Warning, this has not been implemented yet!")
        with open(args.config_runlike, "r") as file:
            runlike_config = yaml.safe_load(file)
        run_likelihood_inference(runlike_config)

    # If neither config is provided, print a message
    if not args.config_train and not args.config_test and not args.config_runlike:
        print("No configuration file provided. Please specify --config-train or --config-test.")
    


def train_likefree_inference(config, overwrite=False):
    """
    Train function using parameters from the config file.
    """

    dir_results = '/scratch/kstoreyf/muchisimocks/results' # hyperion path

    # Read settings from config file
    data_mode = config["data_mode"]
    statistics = config["statistics"]
    # for now before i extend to multiple!
    #statistic = statistics[0]
    n_train = config["n_train"]
    tag_params = config["tag_params"]
    tag_biasparams = config["tag_biasparams"]
    tag_noise = config.get("tag_noise", None)  # noise parameters
    tag_Anoise = config.get("tag_Anoise", None)  # noise parameters
    kwargs_data = config["kwargs_data"]
    run_mode = config["run_mode"]
    sweep_name = config["sweep_name"]
    tag_data = config["tag_data"]
    tag_inf = config["tag_inf"]

    dir_sbi = f'{dir_results}/results_sbi/sbi{tag_inf}'
    fn_posterior = f"{dir_sbi}/posterior.p"
    if not overwrite and os.path.exists(fn_posterior):
        print(f"Oh look, posterior.p already exists in {dir_sbi}, and overwrite={overwrite}! Skipping training.")
        return
    

    ### Load data and parameters
    # don't need the fixed params for training!
    k, y, y_err, idxs_params, params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, Anoise_df, Anoise_dict_fixed, random_ints_cosmo, random_ints_bias = \
                data_loader.load_data(data_mode, statistics, 
                                      tag_params, tag_biasparams,
                                      tag_noise=tag_noise,
                                      tag_Anoise=tag_Anoise,
                                      tag_data=tag_data,
                                      kwargs=kwargs_data)

    # get bounds dict
    _, dict_bounds_cosmo, _ = genp.define_LH_cosmo(tag_params)
    _, dict_bounds_bias, _ = genp.define_LH_bias(tag_biasparams)
    dict_bounds = {**dict_bounds_cosmo, **dict_bounds_bias}
    # add noise parameter bounds if they exist
    if tag_Anoise is not None:
        _, dict_bounds_noise, _ = genp.define_LH_Anoise(tag_Anoise)
        dict_bounds.update(dict_bounds_noise)
    
    ### Subsampling (ntrain and train/val)
    # downsample based on n_train    
    if n_train is None:
        n_train = len(random_ints_cosmo)
    # gets all the random ints less than ntrain, so guarantees we're not missing numbers
    random_ints_cosmo = random_ints_cosmo[random_ints_cosmo<n_train]
    # then these are split fractionally into train and val
    idxs_cosmo_train, idxs_cosmo_val, _ = utils.idxs_train_val_test(random_ints_cosmo, frac_train=0.9, frac_val=0.1, frac_test=0.0)

    # for each row in the index metadata of the full dataset, 
    # if our intended training idx is in it, keep
    # y_shape is (n_stats, n_idxs, n_bins) (inhomogeneous!)
    idxs_all = np.arange(len(y[0]))
    # first column of idxs_params is the cosmo index
    idxs_train = idxs_all[np.isin(idxs_params[:,0], idxs_cosmo_train)]
    idxs_val = idxs_all[np.isin(idxs_params[:,0], idxs_cosmo_val)]

    theta, param_names = data_loader.param_dfs_to_theta(idxs_params, params_df, biasparams_df, Anoise_df,
                                                        n_rlzs_per_cosmo=config["n_rlzs_per_cosmo"])
    print('theta shape:', theta.shape)
    print(param_names)

    theta_train, theta_val = theta[idxs_train], theta[idxs_val]
    y_train, y_val, y_err_train, y_err_val = [], [], [], []
    for i_stat in range(len(statistics)):
        print(f"y train shape for statistic {statistics[i_stat]}:", y[i_stat][idxs_train].shape)
        y_train.append(y[i_stat][idxs_train])
        y_val.append(y[i_stat][idxs_val])
        y_err_train.append(y_err[i_stat][idxs_train])
        y_err_val.append(y_err[i_stat][idxs_val])    
        
    print("y_train shape:", len(y_train), len(y_train[0]), len(y_train[0][0]))
    # y_train, y_val = y[:,idxs_train], y[:,idxs_val]
    # y_err_train, y_err_val = y_err[:,idxs_train], y_err[:,idxs_val]
        
    ### Run inference (now only sbi)
    # Run mode and sweep configuration
    print("tag_inf (SBI):", tag_inf)
    
    sbi_network = sbi_model.SBIModel(
                theta_train=theta_train,
                y_train_unscaled=y_train,
                y_err_train_unscaled=y_err_train,
                theta_val=theta_val,
                y_val_unscaled=y_val,
                y_err_val_unscaled=y_err_val,
                tag_sbi=tag_inf,
                run_mode=run_mode,
                sweep_name=sweep_name,
                param_names=param_names,
                statistics=statistics,
                dict_bounds=dict_bounds,
                )
    sbi_network.run(max_epochs=2000)
    #sbi_network.run(max_epochs=10)


def test_likefree_inference(config, overwrite=False):
    """
    Test function using parameters from the config file."""

    dir_results = '/scratch/kstoreyf/muchisimocks/results' # hyperion path

    # Read settings from config file
    data_mode = config["data_mode"]
    statistics = config["statistics"]
    # for now before i extend to multiple!
    #statistic = statistics[0]
    tag_params = config["tag_params"]
    tag_biasparams = config["tag_biasparams"]
    tag_noise = config.get("tag_noise", None)  
    tag_Anoise = config.get("tag_Anoise", None) 
    evaluate_mean = config["evaluate_mean"]
    idxs_obs = config["idxs_obs"]
    #idxs_obs = np.arange(10)
    #idxs_obs = [0,1,2]
    kwargs_data_test = config["kwargs_data_test"]
    tag_params_test = config["tag_params_test"]
    tag_biasparams_test = config["tag_biasparams_test"]
    tag_noise_test = config.get("tag_noise_test", None) 
    tag_Anoise_test = config.get("tag_Anoise_test", None) 
    tag_data_train = config["tag_data_train"]
    tag_data_test = config["tag_data_test"]
    tag_inf_train = config["tag_inf_train"]
    sweep_name = config["sweep_name"]
    
    if evaluate_mean:
        tag_test = f'{tag_data_test}_mean'
    else:
        tag_test = tag_data_test
    dir_sbi = f'{dir_results}/results_sbi/sbi{tag_inf_train}'
    fn_samples_test_pred = f'{dir_sbi}/samples_test{tag_test}_pred.npy'
    if not overwrite and os.path.exists(fn_samples_test_pred):
        print(f"Oh look, samples {fn_samples_test_pred} already exists, and overwrite={overwrite}! Skipping training.")
        return
    
    print(statistics, tag_params, tag_biasparams)
    print(tag_params_test, tag_biasparams_test)
    # n_rlzs_per_cosmo = kwargs_data_test["n_rlzs_per_cosmo"]
    ### Load data and parameters
    # our setup is such that that the test set is a separate dataset, so no need to split
    # don't need theta either - just predicting, not comparing
    print(tag_noise_test, tag_Anoise_test)
    k, y, y_err, idxs_params, params_df, cosmo_param_dict_fixed, biasparams_df, bias_param_dict_fixed, Anoise_df, Anoise_dict_fixed, random_ints, random_ints_bias = \
                data_loader.load_data(data_mode, statistics,
                                      tag_params_test, tag_biasparams_test,
                                      tag_noise=tag_noise_test,
                                      tag_Anoise=tag_Anoise_test,
                                      tag_data=tag_data_train, #this goes to mask
                                      kwargs=kwargs_data_test)

    param_names_train = data_loader.get_param_names(tag_params=tag_params, tag_biasparams=tag_biasparams, tag_Anoise=tag_Anoise)

    sbi_network = sbi_model.SBIModel(
                tag_sbi=tag_inf_train,
                run_mode='load',
                sweep_name=sweep_name,
                param_names=param_names_train,
                statistics=statistics,
                )
    sbi_network.run() #need this to do the loading
    # TODO make this work for both emu and muchisimocks # ?? not sure what this means rn
    if idxs_obs is None:
        y_obs = y
    else:
        y_obs = []
        for i_stat in range(len(statistics)):
            y_obs.append(y[i_stat][idxs_obs])

    # maybe should load this in as a separate dataset, but for now seems fine to do this way
    if evaluate_mean:
        # make array of arrays-of-means of each stat
        y_mean = []
        for i_stat in range(len(statistics)):
            y_mean_i = np.mean(y[i_stat], axis=0)
            y_mean.append(y_mean_i)
        sbi_network.evaluate_test_set(y_test_unscaled=y_mean, tag_test=tag_test)
    else:
        # run on full test set
        sbi_network.evaluate_test_set(y_test_unscaled=y_obs, tag_test=tag_test)


def run_likelihood_inference(config):
    """
    Run likelihood-based inference using parameters from the config file.
    """
    mcmc_framework = config.get('mcmc_framework', 'dynesty')
    idxs_obs = config.get('idxs_obs', [0])
    evaluate_mean = config.get('evaluate_mean', False)
    data_mode = config['data_mode']
    statistics = config['statistics']
    tag_params = config['tag_params']
    tag_biasparams = config['tag_biasparams']
    tag_Anoise = config.get('tag_Anoise', None)  # noise parameters
    tag_data = config.get('tag_data', None)
    tag_inf = config['tag_inf']
    cosmo_param_names_vary = config.get('cosmo_param_names_vary', [])
    bias_param_names_vary = config.get('bias_param_names_vary', [])
    kwargs_data = config.get('kwargs_data', {})
    assert len(statistics)==1 and statistics[0]=='pk', "Currently only pk is supported for likelihood inference"

    if evaluate_mean:
        assert 'p0' in tag_params, "If you're evaluating the mean, don't you want fixed cosmo?"
    
    # Load data and parameters
    k, y, y_err, idxs_params, params_df, cosmo_param_dict_fixed, biasparams_df, bias_param_dict_fixed, Anoise_df, Anoise_dict_fixed, random_ints, random_ints_bias = \
        data_loader.load_data(data_mode, statistics,
                              tag_params, tag_biasparams, 
                              tag_Anoise=tag_Anoise,
                              tag_data=tag_data, 
                              kwargs=kwargs_data)

    # for now only pk implemented, so just take first
    k = k[0]
    y = y[0]
    y_err = y_err[0]

    if evaluate_mean:
        print("Evaluating mean of test set! idxs_obs ignored")
        # make array of arrays-of-means of each stat
        ys_obs = np.array([np.mean(y, axis=0)])
        ys_err_obs = np.array([np.mean(y_err, axis=0)]) # is it fine to take the mean of the err?
    else:
        # just grab the ones we're going to loop over to observe
        ys_obs = y[idxs_obs]
        ys_err_obs = y_err[idxs_obs]
        
    # get bounds
    _, dict_bounds_cosmo, _ = genp.define_LH_cosmo(tag_params)
    _, dict_bounds_bias, _ = genp.define_LH_bias(tag_biasparams)
    dict_bounds = {**dict_bounds_cosmo, **dict_bounds_bias}
    
    # add noise parameter bounds if they exist
    if tag_Anoise is not None:
        _, dict_bounds_noise, _ = genp.define_LH_Anoise(tag_Anoise)
        dict_bounds.update(dict_bounds_noise)
        
    print("bounds")
    print(dict_bounds_cosmo)
    print(dict_bounds_bias)
    if tag_Anoise is not None:
        print(dict_bounds_noise)

    # not using training data for likelihood methods, so unclear how to scale; 
    # let's just do log for now
    # (maybe really should take reasonable bounds from training data?)
    scaler_y = scl.Scaler('log')
    scaler_y.fit(ys_obs)
    ys_obs_scaled = scaler_y.scale(ys_obs)
    ys_err_obs_scaled = scaler_y.scale_error(ys_err_obs, ys_obs)

    dir_emus_lbias = '/home/kstoreyf/external' #hyperion path
    emu, emu_bounds, emu_param_names_all = utils.load_emu(dir_emus_lbias=dir_emus_lbias)    
    
    for i in range(len(ys_obs)):
        if evaluate_mean:
            idx_obs = 0 #shouldn't matter because should just be using fixed set! 
            tag_obs = '_mean'
        else:
            idx_obs = idxs_obs[i]
            tag_obs = f'_idx{idx_obs}'
        y_obs_unscaled = ys_obs[i]
        y_obs = ys_obs_scaled[i]
        cosmo_param_dict_fixed_obs = cosmo_param_dict_fixed.copy()
        if params_df is not None:
            cosmo_param_dict_fixed_obs.update(params_df.loc[idx_obs].to_dict())
        for pn in cosmo_param_names_vary:
            cosmo_param_dict_fixed_obs.pop(pn, None)
        bias_param_dict_fixed_obs = bias_param_dict_fixed.copy()
        if biasparams_df is not None:
            # TODO this only works in case that biasparams_df is same length as params_df!
            # need to update to get proper idxs_bias
            bias_param_dict_fixed_obs.update(biasparams_df.loc[idx_obs].to_dict())
        for pn in bias_param_names_vary:
            bias_param_dict_fixed_obs.pop(pn, None)
            
        # construct covariance matrix
        err_1p = 0.01*y_obs_unscaled
        err_1p_scaled = scaler_y.scale_error(err_1p, y_obs_unscaled)
        err_gaussian_scaled = ys_err_obs_scaled[i]
        var = err_gaussian_scaled**2 + err_1p_scaled**2
        cov_inv = np.diag(1/var)
        
        import mcmc
        mcmc.evaluate_mcmc(idx_obs, y_obs, cov_inv, scaler_y, 
                emu, k, 
                cosmo_param_dict_fixed_obs, bias_param_dict_fixed_obs, 
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


if __name__=='__main__':
    main()