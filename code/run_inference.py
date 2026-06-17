import os
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
# BX_SWEEP / N_TRAIN_SWEEP: must match generate_config_inference; used for run_mode best (copy vs retrain).
from generate_config_inference import (
    BX_SWEEP,
    N_TRAIN_SWEEP,
    SWEEP_NUM_RUNS,
    build_tag_data,
    tags_mask_for_sweep,
)

# Coverage test-set batching (evaluate_test_set); not written to YAML configs.
TEST_CHECKPOINT_EVERY = 20
TEST_BATCH_TIMEOUT_SECONDS = 3600.0  # 1 h; timed-out batches get NaN placeholders


def _evaluate_test_set_batch_kwargs(evaluate_mean: bool) -> dict:
    """Batch size / timeout for evaluate_test_set (timeout only for multi-mock coverage runs)."""
    return {
        "checkpoint_every": TEST_CHECKPOINT_EVERY,
        "batch_timeout_seconds": None if evaluate_mean else TEST_BATCH_TIMEOUT_SECONDS,
    }


def _build_tags_mask(statistics, config) -> list[str]:
    """
    Require `tags_mask` to be provided in the config.

    `tags_mask` must be a list of strings with length == len(statistics),
    aligned by index (tags_mask[i] applies to statistics[i]).
    """
    if "tags_mask" not in config:
        raise KeyError("Config must include `tags_mask` (a list aligned with `statistics`).")

    tags_mask = config["tags_mask"]
    if isinstance(tags_mask, str):
        raise TypeError("`tags_mask` must be a list (not a single string).")

    tags_mask = list(tags_mask)
    if len(tags_mask) != len(statistics):
        raise ValueError(
            f"tags_mask must have same length as statistics. Got {len(tags_mask)} vs {len(statistics)}."
        )
    return tags_mask


def main():
    
    parser = argparse.ArgumentParser(description="Run inference with config files.")
    parser.add_argument("-tr", "--config-train", type=str, help="Path to the training YAML configuration file.")
    parser.add_argument("-te", "--config-test", type=str, help="Path to the testing YAML configuration file.")
    parser.add_argument("-l", "--config-runlike", type=str, help="Path to the runlike YAML configuration file.")
    parser.add_argument(
        "--overwrite-train",
        action="store_true",
        help="Re-run training even if posterior.p already exists in the output dir.",
    )
    parser.add_argument(
        "--overwrite-test",
        action="store_true",
        help="Re-run testing even if samples_test*_pred.npy already exists.",
    )
    args = parser.parse_args()

    

    # Run training if a training config file is provided
    if args.config_train:
        with open(args.config_train, "r") as file:
            train_config = yaml.safe_load(file)
        train_likefree_inference(
            train_config,
            overwrite=args.overwrite_train,
            config_yaml_path=args.config_train,
        )

    # Run testing if a testing config file is provided
    if args.config_test:
        with open(args.config_test, "r") as file:
            test_config = yaml.safe_load(file)
        data_mode_test_default = "muchisimocks"
        if test_config.get("data_mode_test", data_mode_test_default) == "muchisimocks":
            test_likefree_inference(test_config, overwrite=args.overwrite_test)
        else:
            test_likefree_inference_ood(test_config, overwrite=args.overwrite_test)

    # WARNING not implemented yet !
    if args.config_runlike:
        print("Warning, this has not been implemented yet!")
        with open(args.config_runlike, "r") as file:
            runlike_config = yaml.safe_load(file)
        run_likelihood_inference(runlike_config)

    # If neither config is provided, print a message
    if not args.config_train and not args.config_test and not args.config_runlike:
        print("No configuration file provided. Please specify --config-train or --config-test.")
    


def train_likefree_inference(config, overwrite=False, config_yaml_path=None):
    """
    Train function using parameters from the config file.

    ``config_yaml_path``: path from ``--config-train``; used to write
    ``wandb_sweep_id`` when a new W&B sweep is created (sweep mode).
    """

    dir_results = str(paths.DIR_RESULTS)  # default from paths.py; override via env if needed

    # Read settings from config file
    data_mode = config["data_mode"]
    statistics = config["statistics"]
    # for now before i extend to multiple!
    #statistic = statistics[0]
    n_train = config["n_train"]
    tag_params = config["tag_params"]
    tag_biasparams = config["tag_biasparams"]
    tag_noise = config.get("tag_noise", None)  # noise parameters
    run_mode = config["run_mode"]
    sweep_name = config["sweep_name"]
    tag_inf = config["tag_inf"]
    bx = config.get("bx", None)
    tags_mask = _build_tags_mask(statistics, config)

    dir_sbi = f'{dir_results}/results_sbi/sbi{tag_inf}'
    fn_posterior = f"{dir_sbi}/posterior.p"
    if not overwrite and os.path.exists(fn_posterior):
        print(f"Oh look, posterior.p already exists in {dir_sbi}, and overwrite={overwrite}! Skipping training.")
        return
    
    print(f"Training with tag_params={tag_params}, tag_biasparams={tag_biasparams}, tag_noise={tag_noise}")

    ### Load data and parameters
    # don't need the fixed params for training!
    k, y, y_err, idxs_params, params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, random_ints_cosmo, random_ints_bias = \
                data_loader.load_data(data_mode, statistics, 
                                      tag_params, tag_biasparams,
                                      tag_noise=tag_noise,
                                      tags_mask=tags_mask,
                                      bx=bx)
    
    # turn parameters into nice theta for training
    theta, param_names = data_loader.param_dfs_to_theta(
        idxs_params, params_df, biasparams_df
    )

    # Build bounds directly from the known parameter set bounds.
    # Noise parameters are stored inside tag_biasparams now.
    dict_bounds = {}
    for pn in param_names:
        for bounds_set in genp.BOUNDS.values():
            if pn in bounds_set:
                dict_bounds[pn] = bounds_set[pn]
                break
        else:
            raise KeyError(f"Missing bounds for parameter '{pn}'. Update generate_params.BOUNDS.")
    print('theta shape:', theta.shape)
    print(param_names)
    
    # Reparameterize if requested
    reparameterize = config.get("reparameterize", False)
    plite = config.get("plite", False)
    if reparameterize:
        print("For now setting plite=True whenever reparameterize is true")
        plite = True

    if reparameterize:
        print("Reparameterizing theta...")
        theta, param_names = utils_inference.reparameterize_theta(theta, param_names)
        # Also update bounds for reparameterized parameters
        dict_bounds = utils_inference.reparameterize_bounds(dict_bounds)
        print('theta shape after reparameterization:', theta.shape)
        print('param_names after reparameterization:', param_names)
        print('Updated bounds:', dict_bounds)
    
    # If plite is true, exclude hubble, omega_baryon, ns from training
    if plite:
        print("plite=True: Excluding hubble, omega_baryon, ns from training")
        params_to_exclude = ['hubble', 'omega_baryon', 'ns']
        
        # Find indices of parameters to exclude
        idxs_to_exclude = [i for i, pn in enumerate(param_names) if pn in params_to_exclude]
        
        if idxs_to_exclude:
            print(f"Excluding parameters: {[param_names[i] for i in idxs_to_exclude]}")
            # Filter theta
            idxs_to_keep = [i for i in range(len(param_names)) if i not in idxs_to_exclude]
            theta = theta[:, idxs_to_keep]
            # Filter param_names
            param_names = [param_names[i] for i in idxs_to_keep]
            # Filter dict_bounds
            dict_bounds = {pn: bounds for pn, bounds in dict_bounds.items() if pn not in params_to_exclude}
            print('theta shape after plite filtering:', theta.shape)
            print('param_names after plite filtering:', param_names)
            print('Updated bounds after plite filtering:', dict_bounds)
    
    ### Subsampling (n_train); sbi does train/val split internally via validation_fraction
    if n_train is None:
        n_train = len(random_ints_cosmo)
    n_train_used = int(n_train)
    idxs_cosmo_subset = random_ints_cosmo[:n_train]

    # for each row in the index metadata of the full dataset,
    # if our intended training idx is in it, keep
    # y_shape is (n_stats, n_idxs, n_bins) (inhomogeneous!)
    idxs_all = np.arange(len(y[0]))
    # first column of idxs_params is the cosmo index
    idxs_train = idxs_all[np.isin(idxs_params[:, 0], idxs_cosmo_subset)]

    theta_train = theta[idxs_train]
    y_train = []
    for i_stat in range(len(statistics)):
        print(f"y train shape for statistic {statistics[i_stat]}:", y[i_stat][idxs_train].shape)
        y_train.append(y[i_stat][idxs_train])

    print("y_train shape:", len(y_train), len(y_train[0]), len(y_train[0][0]))
        
    ### Run inference (now only sbi)
    print("tag_inf (SBI):", tag_inf)
    # run_mode best: copy sweep posterior only if bx/n_train AND tags_mask match the sweep
    # (fiducial masks from tags_mask_for_sweep). Otherwise retrain with best hparams —
    # e.g. training _kb0.2 while sweep used TAG_MASK_BISPEC_SWEEP _kb0.25.
    sweep_tags_mask = tags_mask_for_sweep(statistics)
    mask_matches_sweep = list(tags_mask) == list(sweep_tags_mask)
    matches_sweep_model = (
        bx is not None
        and int(bx) == BX_SWEEP
        and int(n_train) == N_TRAIN_SWEEP
        and mask_matches_sweep
    )
    tag_sweep = config.get("tag_sweep")
    if run_mode == "best" and tag_sweep and sweep_name:
        reparameterize = config.get("reparameterize", False)
        tag_data_sweep = build_tag_data(
            data_mode,
            statistics,
            sweep_tags_mask,
            tag_params,
            tag_biasparams,
            tag_noise,
        )
        default_sweep_name = (
            tag_data_sweep
            + ("_rp" if reparameterize else "")
            + f"_bx{BX_SWEEP}_ntrain{N_TRAIN_SWEEP}_sweep{tag_sweep}"
        )
        if sweep_name != default_sweep_name:
            print(
                "run_mode=best: sweep_name differs from default for this train bundle "
                f"(sweep_name={sweep_name!r}, default={default_sweep_name!r}) — "
                "retrain with best hparams only (no checkpoint copy).",
                flush=True,
            )
            matches_sweep_model = False
    if run_mode == "best":
        print(
            "run_mode=best: tags_mask=%s | fiducial sweep masks=%s | mask_matches_sweep=%s"
            % (tags_mask, sweep_tags_mask, mask_matches_sweep),
            flush=True,
        )
        print(
            "run_mode=best: matches_sweep_model=%s (if True: copy sweep posterior; if False: retrain with best hparams)"
            % matches_sweep_model,
            flush=True,
        )
        print(
            "run_mode=best: bx/n_train=%s/%s (sweep fiducial %s/%s)"
            % (bx, n_train, BX_SWEEP, N_TRAIN_SWEEP),
            flush=True,
        )

    sweep_num_runs = int(config["sweep_num_runs"]) if run_mode == "sweep" else SWEEP_NUM_RUNS

    sbi_network = sbi_model.SBIModel(
                theta_train=theta_train,
                y_train_unscaled=y_train,
                tag_sbi=tag_inf,
                run_mode=run_mode,
                sweep_name=sweep_name,
                param_names=param_names,
                statistics=statistics,
                dict_bounds=dict_bounds,
                overwrite=overwrite,
                matches_sweep_model=matches_sweep_model,
                wandb_sweep_id=config.get("wandb_sweep_id"),
                sweep_num_runs=sweep_num_runs,
                wandb_config_yaml_path=config_yaml_path,
                )
    sbi_network.run(max_epochs=2000)
    #sbi_network.run(max_epochs=10)


def test_likefree_inference(config, overwrite=False):
    """
    Test function using parameters from the config file."""

    dir_results = str(paths.DIR_RESULTS)

    # Read settings from config file
    data_mode = config["data_mode"]
    statistics = config["statistics"]
    # for now before i extend to multiple!
    #statistic = statistics[0]
    tag_params = config["tag_params"]
    tag_biasparams = config["tag_biasparams"]
    tag_noise = config.get("tag_noise", None)  
    evaluate_mean = config["evaluate_mean"]
    idxs_obs = config["idxs_obs"]
    #idxs_obs = np.arange(10)
    #idxs_obs = [0,1,2]
    tag_params_test = config["tag_params_test"]
    tag_biasparams_test = config["tag_biasparams_test"]
    tag_noise_test = config.get("tag_noise_test", None) 
    tag_data_train = config["tag_data_train"]
    tag_data_test = config["tag_data_test"]
    tag_inf_train = config["tag_inf_train"]
    n_test_eval = config.get("n_test_eval", None)
    tags_mask = _build_tags_mask(statistics, config)
    batch_kwargs = _evaluate_test_set_batch_kwargs(evaluate_mean)
    #print("BEWARNED: manually setting n_test_eval to 100")
    #n_test_eval = 100
    
    if evaluate_mean:
        tag_test = f'{tag_data_test}_mean'
    else:
        tag_test = tag_data_test
    
    # Construct tag_test_eval with _N{n_test_eval} suffix if n_test_eval is specified
    if n_test_eval is not None:
        tag_n_eval = f"_neval{n_test_eval}"
        tag_test_eval = f"{tag_test}{tag_n_eval}"
    else:
        tag_test_eval = tag_test
    
    dir_sbi = f'{dir_results}/results_sbi/sbi{tag_inf_train}'
    
    # Check if file already exists (using tag_test_eval if provided, otherwise tag_test)
    fn_samples_test_pred = f'{dir_sbi}/samples_test{tag_test_eval}_pred.npy'
    if not overwrite and os.path.exists(fn_samples_test_pred):
        print(f"Oh look, samples {fn_samples_test_pred} already exists, and overwrite={overwrite}! Skipping testing.")
        return
    
    print(statistics, tag_params, tag_biasparams)
    print(tag_params_test, tag_biasparams_test)
    ### Load data and parameters
    # our setup is such that that the test set is a separate dataset, so no need to split
    # don't need theta either - just predicting, not comparing
    print(tag_noise_test)
    k, y, y_err, idxs_params, params_df, cosmo_param_dict_fixed, biasparams_df, bias_param_dict_fixed, random_ints, random_ints_bias = \
                data_loader.load_data(data_mode, statistics,
                                      tag_params_test, tag_biasparams_test,
                                      tag_noise=tag_noise_test,
                                      tags_mask=tags_mask,
                                      )

    param_names_train = data_loader.get_param_names(tag_params=tag_params, tag_biasparams=tag_biasparams)

    sbi_network = sbi_model.SBIModel(
                tag_sbi=tag_inf_train,
                run_mode='load',
                param_names=param_names_train,
                statistics=statistics,
                overwrite=overwrite,
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
        sbi_network.evaluate_test_set(
            y_test_unscaled=y_mean,
            tag_test_eval=tag_test_eval,
            n_test_eval=n_test_eval,
            **batch_kwargs,
        )
    else:
        # run on full test set
        print(f"y_obs shape: {len(y_obs)}, {len(y_obs[0])}, {len(y_obs[0][0])}")
        sbi_network.evaluate_test_set(
            y_test_unscaled=y_obs,
            tag_test_eval=tag_test_eval,
            n_test_eval=n_test_eval,
            **batch_kwargs,
        )


def test_likefree_inference_ood(config, overwrite=False):
    """
    Test function using parameters from the config file."""

    dir_results = str(paths.DIR_RESULTS)

    # Read settings from config file
    # data_mode is for TEST DATA; muchisimocks, emu, shame, etc

    data_mode = config["data_mode"]
    statistics = config["statistics"]
    # for now before i extend to multiple!
    #statistic = statistics[0]
    ### training 
    tag_params = config["tag_params"]
    tag_biasparams = config["tag_biasparams"]
    tag_noise = config.get("tag_noise", None)  
    tag_data_train = config["tag_data_train"]
    tag_inf_train = config["tag_inf_train"]
    ### testing
    data_mode_test = config["data_mode_test"]
    evaluate_mean = config["evaluate_mean"]
    idxs_obs = config["idxs_obs"]
    tag_mock = config['tag_mock']
    tag_data_test = config["tag_data_test"]
    n_test_eval = config.get("n_test_eval", None)
    tags_mask = _build_tags_mask(statistics, config)
    batch_kwargs = _evaluate_test_set_batch_kwargs(evaluate_mean)
    
    if evaluate_mean:
        tag_test = f'{tag_data_test}_mean'
    else:
        tag_test = tag_data_test
    
    # Construct tag_test_eval with _neval{n_test_eval} suffix if n_test_eval is specified
    if n_test_eval is not None:
        tag_n_eval = f"_neval{n_test_eval}"
        tag_test_eval = f"{tag_test}{tag_n_eval}"
    else:
        tag_test_eval = tag_test
    
    dir_sbi = f'{dir_results}/results_sbi/sbi{tag_inf_train}'
    
    # Check if file already exists (using tag_test_eval)
    fn_samples_test_pred = f'{dir_sbi}/samples_test{tag_test_eval}_pred.npy'
    if not overwrite and os.path.exists(fn_samples_test_pred):
        print(f"Oh look, samples {fn_samples_test_pred} already exists, and overwrite={overwrite}! Skipping testing.")
        return
    
    print(statistics, tag_params, tag_biasparams)
                
    # tag_data_train goes to mask, NOTE this could be structured more clearly...
    k, y, y_err = data_loader.load_data_ood(data_mode_test, statistics, tag_mock, tags_mask=tags_mask)

    param_names_train = data_loader.get_param_names(tag_params=tag_params, tag_biasparams=tag_biasparams)

    sbi_network = sbi_model.SBIModel(
                tag_sbi=tag_inf_train,
                run_mode='load',
                param_names=param_names_train,
                statistics=statistics,
                overwrite=overwrite,
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
        sbi_network.evaluate_test_set(
            y_test_unscaled=y_mean,
            tag_test_eval=tag_test_eval,
            n_test_eval=n_test_eval,
            **batch_kwargs,
        )
    else:
        # run on full test set
        #print(f"y_obs shape: {len(y_obs)}, {len(y_obs[0])}, {len(y_obs[0][0])}")
        sbi_network.evaluate_test_set(
            y_test_unscaled=y_obs,
            tag_test_eval=tag_test_eval,
            n_test_eval=n_test_eval,
            **batch_kwargs,
        )


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
        
    # get bounds
    _, dict_bounds_cosmo, _ = genp.define_LH_cosmo(tag_params)
    _, dict_bounds_bias, _ = genp.define_LH_bias(tag_biasparams)
    dict_bounds = {**dict_bounds_cosmo, **dict_bounds_bias}
    
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
           

if __name__=='__main__':
    main()