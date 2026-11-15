"""Functions moved out of code/run_inference.py in the v2 cleanup (2026-11-15).

No longer used by the pipeline; kept for reference / possible reuse, not maintained.
Imports are copied from code/run_inference.py; module-level names come from the star import.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import yaml
import paths
import utils_inference
import utils_model
import sbi_model
import scaler_custom as scl
import data_loader
import generate_params as genp
from generate_config_inference import (
    BX_SWEEP,
    N_TRAIN_SWEEP,
    SWEEP_NUM_RUNS,
    tags_mask_for_sweep,
)
from run_inference import *  # noqa: F401,F403


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
    tag_inf = config['tag_inf']
    cosmo_param_names_vary = config.get('cosmo_param_names_vary', [])
    bias_param_names_vary = config.get('bias_param_names_vary', [])
    assert len(statistics)==1 and statistics[0]=='pk', "Currently only pk is supported for likelihood inference"

    if evaluate_mean:
        assert 'p0' in tag_params, "If you're evaluating the mean, don't you want fixed cosmo?"
    
    # Load data and parameters
    tags_mask = _build_tags_mask(statistics, config)

    k, y, y_err, idxs_params, params_df, cosmo_param_dict_fixed, biasparams_df, bias_param_dict_fixed, random_ints, random_ints_bias = \
        data_loader.load_data(data_mode, statistics,
                              tag_params, tag_biasparams,
                              tags_mask=tags_mask,
                              )

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
        
    # get bounds (define_LH_cosmo/define_LH_bias were removed; use BOUNDS via get_bounds)
    dict_bounds_cosmo = genp.get_bounds("cosmo")
    dict_bounds_bias = genp.get_bounds("bias")
    
    print("bounds")
    print(dict_bounds_cosmo)
    print(dict_bounds_bias)

    # not using training data for likelihood methods, so unclear how to scale; 
    # let's just do log for now
    # (maybe really should take reasonable bounds from training data?)
    scaler_y = scl.Scaler('log')
    scaler_y.fit(ys_obs)
    ys_obs_scaled = scaler_y.scale(ys_obs)
    ys_err_obs_scaled = scaler_y.scale_error(ys_err_obs, ys_obs)

    dir_emus_lbias = '/home/kstoreyf/external' #hyperion path
    emu, emu_bounds, emu_param_names_all = utils_model.load_emu(dir_emus_lbias=dir_emus_lbias)
    
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


