import os
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse

import numpy as np
import yaml

import paths
import utils_inference
import utils_model
import sbi_model
import data_loader
import generate_params as genp
# BX_SWEEP / N_TRAIN_SWEEP: must match generate_config_inference; used for run_mode best (copy vs retrain).
from generate_config_inference import (
    BX_SWEEP,
    N_TRAIN_SWEEP,
    SWEEP_NUM_RUNS,
    build_sweep_name,
    tags_mask_for_sweep,
)
import torch

# Importing bacco (via generate_params) resets OMP_NUM_THREADS to the node's CPU count (48), so torch would
# sample with 48 threads on a 1-CPU job (~13x slower). Training sets its own count in SBIModel.fit.
torch.set_num_threads(1)

# Coverage test-set batching (evaluate_test_set); not written to YAML configs.
TEST_CHECKPOINT_EVERY = 5
TEST_BATCH_TIMEOUT_SECONDS_DEFAULT = 7200.0  # 2 h; timed-out batches get NaN placeholders


def _evaluate_test_set_batch_kwargs(
    evaluate_mean: bool, batch_timeout_seconds=None, checkpoint_every=None,
) -> dict:
    """Batch size / timeout for evaluate_test_set (no timeout when evaluate_mean)."""
    if batch_timeout_seconds is None:
        batch_timeout_seconds = TEST_BATCH_TIMEOUT_SECONDS_DEFAULT
    if checkpoint_every is None:
        checkpoint_every = TEST_CHECKPOINT_EVERY
    return {
        "checkpoint_every": int(checkpoint_every),
        "batch_timeout_seconds": None if evaluate_mean else float(batch_timeout_seconds),
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
    parser.add_argument(
        "--batch-timeout-seconds",
        type=float,
        default=TEST_BATCH_TIMEOUT_SECONDS_DEFAULT,
        help=(
            "Kill a stalled evaluate_test_set batch after this many seconds and "
            f"fill NaN placeholders (default: {TEST_BATCH_TIMEOUT_SECONDS_DEFAULT:.0f}). "
            "Ignored when evaluate_mean is true."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=TEST_CHECKPOINT_EVERY,
        help=(
            "Number of test observations per evaluate_test_set batch "
            f"(default: {TEST_CHECKPOINT_EVERY})."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Posterior draws per test observation (default: 10000).",
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
            test_likefree_inference(
                test_config,
                overwrite=args.overwrite_test,
                batch_timeout_seconds=args.batch_timeout_seconds,
                checkpoint_every=args.checkpoint_every,
                n_samples=args.n_samples,
            )
        else:
            test_likefree_inference_ood(
                test_config,
                overwrite=args.overwrite_test,
                batch_timeout_seconds=args.batch_timeout_seconds,
                checkpoint_every=args.checkpoint_every,
                n_samples=args.n_samples,
            )

    # If neither config is provided, print a message
    if not args.config_train and not args.config_test:
        print("No configuration file provided. Please specify --config-train or --config-test.")
    


def training_prior_bounds(tag_biasparams):
    """Cosmology bounds + the bias(/noise) bounds of the registered set ``tag_biasparams``."""
    key = tag_biasparams.lstrip("_")
    cfg = genp.PARAM_SETS_NESTED.get(key) or genp.PARAM_SETS_LH.get(key)
    if cfg is None:
        raise KeyError(f"{key!r} is not in generate_params.PARAM_SETS_NESTED / PARAM_SETS_LH")
    bounds = dict(genp.get_bounds("cosmo"))
    bounds.update(genp.get_bounds(cfg["bounds_type"], anoise_option=cfg.get("anoise_option")))
    return bounds


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

    # Prior bounds from the training sets' own registry entries (a parameter like An_homog has
    # different bounds in different noise models, so never look it up across all of BOUNDS).
    all_bounds = training_prior_bounds(tag_biasparams)
    missing = [pn for pn in param_names if pn not in all_bounds]
    if missing:
        raise KeyError(f"No bounds for {missing} in the {tag_biasparams!r} parameter set")
    dict_bounds = {pn: all_bounds[pn] for pn in param_names}
    print('theta shape:', theta.shape)
    print(param_names)
    
    # Reparameterize if requested
    reparameterize = config.get("reparameterize", False)
    # plite (drop hubble, omega_baryon, ns) defaults to on with reparameterize; `plite: false` keeps all
    # cosmo params (2026-11 test of whether predicting more params changes the posteriors)
    plite = config.get("plite", reparameterize)

    # Physical bounds define the posterior support even when training on reparameterized columns.
    dict_bounds_physical = dict(dict_bounds)
    if reparameterize:
        print("Reparameterizing theta...")
        theta, param_names = utils_inference.reparameterize_theta(theta, param_names)
        # Also update bounds for reparameterized parameters
        dict_bounds = utils_inference.reparameterize_bounds(dict_bounds)
        print('theta shape after reparameterization:', theta.shape)
        print('param_names after reparameterization:', param_names)
        print('Updated bounds:', dict_bounds)
    
    # If plite is true, exclude hubble, omega_baryon, ns from training
    # exclude_noise (temporary, for the 2026-11 with/without-noise comparison): also drop the noise
    # amplitudes, so the network learns the posterior marginalized over them (their training prior)
    exclude_noise = config.get("exclude_noise", False)
    if plite or exclude_noise:
        params_to_exclude = ['hubble', 'omega_baryon', 'ns'] if plite else []
        if exclude_noise:
            noise_names = list(utils_model.noiseparam_names_ordered)
            params_to_exclude += noise_names + [utils_inference.reparameterized_sample_name(n) for n in noise_names]
        print(f"Excluding from training (plite={plite}, exclude_noise={exclude_noise}): {params_to_exclude}")

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
    # run_mode best: copy the sweep trial's posterior only if this config would train exactly the
    # model the sweep trained (same data, cuts, bx/n_train, seed 0, default parameter set);
    # otherwise retrain with its hyperparameters on this config's data.
    seed = int(config.get("seed", 0))
    sweep_tags_mask = tags_mask_for_sweep(statistics)
    own_sweep_name = build_sweep_name(data_mode, statistics, tag_params, tag_biasparams, tag_noise,
                                      reparameterize, config.get("tag_sweep"))
    copy_checks = {
        "bx": bx is not None and int(bx) == BX_SWEEP,
        "n_train": int(n_train) == N_TRAIN_SWEEP,
        "tags_mask": list(tags_mask) == list(sweep_tags_mask),
        "sweep_data": sweep_name == own_sweep_name,  # sweep was trained on this config's data
        "seed": seed == 0,  # sweep trials use seed 0
        "params": plite == reparameterize and not exclude_noise,  # sweep's default parameter set
    }
    matches_sweep_model = all(copy_checks.values())
    if run_mode == "best":
        print(
            "run_mode=best: matches_sweep_model=%s (True: copy sweep posterior; False: retrain with "
            "best hparams); checks=%s; sweep_name=%s; own sweep name=%s"
            % (matches_sweep_model, copy_checks, sweep_name, own_sweep_name),
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
                dict_bounds_physical=dict_bounds_physical,
                overwrite=overwrite,
                matches_sweep_model=matches_sweep_model,
                wandb_sweep_id=config.get("wandb_sweep_id"),
                sweep_num_runs=sweep_num_runs,
                wandb_config_yaml_path=config_yaml_path,
                nth_best_run=config.get("nth_best_run"),
                sweep_run_id=config.get("sweep_run_id"),
                groups_train=idxs_params[idxs_train, 0],  # cosmology index per row -> grouped val split
                seed=int(config.get("seed", 0)),
                )
    sbi_network.run(max_epochs=2000)
    if sbi_network.training_incomplete:
        # stopped before MUCHISIMOCKS_TRAIN_DEADLINE with a checkpoint; the job requeues itself
        print(f"Training incomplete (checkpointed); exiting with code {sbi_model.EXIT_TRAINING_INCOMPLETE}")
        sys.exit(sbi_model.EXIT_TRAINING_INCOMPLETE)
    #sbi_network.run(max_epochs=10)


def test_likefree_inference(
    config, overwrite=False, batch_timeout_seconds=None, checkpoint_every=None,
    n_samples=10000,
):
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
    # Named subset of test rows: data/params/idxs_obs{tag_params_test}_{idxs_obs_subset}.npy holds
    # param-set indices (e.g. code_auxiliary/select_coverage_subset.py); samples get tag _{subset}.
    idxs_obs_subset = config.get("idxs_obs_subset", None)
    assert idxs_obs is None or idxs_obs_subset is not None, \
        "idxs_obs without idxs_obs_subset would write the full-set samples filename"
    tag_params_test = config["tag_params_test"]
    tag_biasparams_test = config["tag_biasparams_test"]
    tag_noise_test = config.get("tag_noise_test", None) 
    tag_data_train = config["tag_data_train"]
    tag_data_test = config["tag_data_test"]
    tag_inf_train = config["tag_inf_train"]
    n_test_eval = config.get("n_test_eval", None)
    tags_mask = _build_tags_mask(statistics, config)
    batch_kwargs = _evaluate_test_set_batch_kwargs(
        evaluate_mean,
        batch_timeout_seconds=batch_timeout_seconds,
        checkpoint_every=checkpoint_every,
    )
    #print("BEWARNED: manually setting n_test_eval to 100")
    #n_test_eval = 100
    
    if evaluate_mean:
        tag_test = f'{tag_data_test}_mean'
    else:
        tag_test = tag_data_test
    if idxs_obs_subset is not None:
        assert not evaluate_mean, "idxs_obs_subset with evaluate_mean is not supported"
        tag_test = f'{tag_test}_{idxs_obs_subset}'

    # Construct tag_test_eval with _N{n_test_eval} suffix if n_test_eval is specified
    if n_test_eval is not None:
        tag_n_eval = f"_neval{n_test_eval}"
        tag_test_eval = f"{tag_test}{tag_n_eval}"
    else:
        tag_test_eval = tag_test

    dir_sbi = f'{dir_results}/results_sbi/sbi{tag_inf_train}'

    # Skip only when the FINAL samples file exists, is fully usable (no NaN
    # placeholders), and there is no in-progress file. NaN holes / short
    # in-progress arrays are retried by evaluate_test_set.
    fn_samples_test_pred = f'{dir_sbi}/samples_test{tag_test_eval}_pred.npy'
    fn_samples_test_pred_inprogress = f'{dir_sbi}/samples_test{tag_test_eval}_pred_inprogress.npy'
    if (
        not overwrite
        and os.path.exists(fn_samples_test_pred)
        and not os.path.exists(fn_samples_test_pred_inprogress)
    ):
        existing_arr = np.load(fn_samples_test_pred, mmap_mode="r")
        if sbi_model.SBIModel.samples_array_fully_usable(existing_arr):
            print(
                f"Oh look, samples {fn_samples_test_pred} already exist and are fully "
                f"usable (shape {existing_arr.shape}), and overwrite={overwrite}! "
                f"Skipping testing."
            )
            return
        n_stored = sbi_model.SBIModel._n_test_obs_in_samples(existing_arr)
        n_pending = len(
            sbi_model.SBIModel.pending_obs_indices(existing_arr, n_stored)
        )
        print(
            f"Found existing samples {fn_samples_test_pred} with {n_pending}/{n_stored} "
            f"NaN/unusable obs slot(s) (shape {existing_arr.shape}) — will retry those."
        )

    print(statistics, tag_params, tag_biasparams)
    print(tag_params_test, tag_biasparams_test)
    print(f"tag_test_eval: {tag_test_eval}")
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
    if idxs_obs_subset is not None:
        fn_idxs = paths.PROJECT_ROOT / "data" / "params" / f"idxs_obs{tag_params_test}_{idxs_obs_subset}.npy"
        idxs_subset = np.load(fn_idxs)
        # param-set index -> data row (rows can differ from param indices if any mock is missing)
        rows_by_param = {int(ic): r for r, ic in enumerate(idxs_params[:, 0])}
        assert len(rows_by_param) == len(idxs_params), "subset selection assumes one row per cosmology"
        missing = [int(i) for i in idxs_subset if int(i) not in rows_by_param]
        assert not missing, f"subset rows missing from the loaded test data: {missing}"
        idxs_obs = [rows_by_param[int(i)] for i in idxs_subset]
        print(f"idxs_obs_subset={idxs_obs_subset}: {len(idxs_obs)} rows from {fn_idxs}")
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
            n_samples=n_samples,
            **batch_kwargs,
        )
    else:
        # run on full test set
        print(f"y_obs shape: {len(y_obs)}, {len(y_obs[0])}, {len(y_obs[0][0])}")
        sbi_network.evaluate_test_set(
            y_test_unscaled=y_obs,
            tag_test_eval=tag_test_eval,
            n_test_eval=n_test_eval,
            n_samples=n_samples,
            **batch_kwargs,
        )


def test_likefree_inference_ood(
    config, overwrite=False, batch_timeout_seconds=None, checkpoint_every=None,
    n_samples=10000,
):
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
    batch_kwargs = _evaluate_test_set_batch_kwargs(
        evaluate_mean,
        batch_timeout_seconds=batch_timeout_seconds,
        checkpoint_every=checkpoint_every,
    )
    
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
    
    # Skip only when the FINAL samples file exists, is fully usable (no NaN
    # placeholders), and there is no in-progress file. NaN holes are retried
    # by evaluate_test_set.
    fn_samples_test_pred = f'{dir_sbi}/samples_test{tag_test_eval}_pred.npy'
    fn_samples_test_pred_inprogress = f'{dir_sbi}/samples_test{tag_test_eval}_pred_inprogress.npy'
    if (
        not overwrite
        and os.path.exists(fn_samples_test_pred)
        and not os.path.exists(fn_samples_test_pred_inprogress)
    ):
        existing_arr = np.load(fn_samples_test_pred, mmap_mode="r")
        if sbi_model.SBIModel.samples_array_fully_usable(existing_arr):
            print(
                f"Oh look, samples {fn_samples_test_pred} already exist and are fully "
                f"usable (shape {existing_arr.shape}), and overwrite={overwrite}! "
                f"Skipping testing."
            )
            return
        n_stored = sbi_model.SBIModel._n_test_obs_in_samples(existing_arr)
        n_pending = len(
            sbi_model.SBIModel.pending_obs_indices(existing_arr, n_stored)
        )
        print(
            f"Found existing samples {fn_samples_test_pred} with {n_pending}/{n_stored} "
            f"NaN/unusable obs slot(s) (shape {existing_arr.shape}) — will retry those."
        )
    
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
            n_samples=n_samples,
            **batch_kwargs,
        )
    else:
        # run on full test set
        #print(f"y_obs shape: {len(y_obs)}, {len(y_obs[0])}, {len(y_obs[0][0])}")
        sbi_network.evaluate_test_set(
            y_test_unscaled=y_obs,
            tag_test_eval=tag_test_eval,
            n_test_eval=n_test_eval,
            n_samples=n_samples,
            **batch_kwargs,
        )


           

if __name__=='__main__':
    main()