import os
import yaml
from pathlib import Path

import utils_model

'''
Generates YAML configuration files for inference.
'''

# Fiducial bx / n_train used to name the wandb sweep (sweep_name) and for
# run_mode `best`: if (bx, n_train) in the config match these, the best run
# artifact is copied; otherwise best hyperparameters are taken from the sweep
# and the model is retrained on the config's bx / n_train.
# Training data always uses config ``bx`` and ``n_train`` (including run_mode
# ``sweep``); only hyperparameters come from the wandb sweep.
BX_SWEEP = 32
N_TRAIN_SWEEP = 10000

# Resolve config output directories relative to repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS_TRAIN_DIR = REPO_ROOT / "configs" / "configs_train"
DEFAULT_CONFIGS_TEST_DIR = REPO_ROOT / "configs" / "configs_test"
DEFAULT_CONFIGS_RUNLIKE_DIR = REPO_ROOT / "configs" / "configs_runlike"


def main():
    overwrite = False
    stat_arr = [
        ['pk'],
        ['pk', 'pgm'],
        ['pk', 'bispec'],
        ['pk', 'bispec', 'pgm'],
    ]
    n_train_arr = [10000]
    #n_train_arr = [500, 1000, 2000, 4000, 6000, 8000, 10000]
    bx_arr = [32]
    for statistics in stat_arr:
        for n_train in n_train_arr:
            for bx in bx_arr:
                generate_train_config(overwrite=overwrite, statistics=statistics, n_train=n_train, bx=bx)
                #generate_test_config(overwrite=overwrite, statistics=statistics, n_train=n_train, bx=bx)
                #generate_test_config_ood(overwrite=overwrite, statistics=statistics, n_train=n_train, bx=bx)
    #generate_runlike_config(overwrite=overwrite)
    
    
def generate_train_config(dir_config=str(DEFAULT_CONFIGS_TRAIN_DIR),
                          overwrite=False,
                          statistics=['pk'],
                          n_train=N_TRAIN_SWEEP, bx=BX_SWEEP):
    """
    Generates a YAML configuration file for training.
    """
    data_mode = 'muchisimocks'
    tag_params = '_p5_n10000'
    tag_biasparams = '_biasnest_p4_n320000'
    tag_noise = None
    #tag_biasparams = '_biasnoisenest_p9_n320000'
    #tag_noise = '_noise_unit_p5_n10000'
    tag_mask = ''
    #tag_mask = '_kb0.25'
    # bx is bias parameters per cosmo (1x, 2x, 4x, 8x, 16x, 32x)

    # tags_mask is aligned with `statistics` (same order).
    # For now the only active mask is the k-bispec cutoff encoded by `_kb0.25`.
    tags_mask = [tag_mask if stat == 'bispec' else '' for stat in statistics]
    tag_masks = ''.join(tags_mask)

    # Inference params: run_mode 'single' | 'sweep' | 'best'; tag_sweep required for sweep/best
    reparameterize = True
    run_mode = 'single'
    tag_sweep = None  # e.g. '-rand10' for sweep/best

    tag_stats = f'_{"_".join(statistics)}'
    tag_paramsall = tag_params + tag_biasparams
    if tag_noise is not None:
        tag_paramsall += tag_noise
    tag_data = '_'+data_mode + tag_stats + tag_masks + tag_paramsall

    tag_inf_num = f'_bx{bx}_ntrain{n_train}'
    base_inf = tag_data + ('_rp' if reparameterize else '') + tag_inf_num
    tag_inf_num_sweep = f'_bx{BX_SWEEP}_ntrain{N_TRAIN_SWEEP}'
    base_inf_sweep = tag_data + ('_rp' if reparameterize else '') + tag_inf_num_sweep

    if run_mode == 'sweep':
        tag_inf = base_inf_sweep + f'_sweep{tag_sweep}'
        sweep_name = base_inf_sweep + f'_sweep{tag_sweep}'
    elif run_mode == 'best':
        tag_inf = base_inf + f'_best{tag_sweep}'
        sweep_name = base_inf_sweep + f'_sweep{tag_sweep}'
    else:  # single
        tag_inf = base_inf
        sweep_name = None

    config = {
        "data_mode": data_mode,
        "statistics": statistics,
        "tag_params": tag_params,
        "tag_biasparams": tag_biasparams,
        "tag_noise": tag_noise,
        "tags_mask": tags_mask,
        "n_train": n_train,
        "bx": bx,
        "run_mode": run_mode,
        "tag_sweep": tag_sweep,
        "sweep_name": sweep_name,
        "tag_data": tag_data,
        "tag_inf": tag_inf,
        "reparameterize": reparameterize,
    }
            
    os.makedirs(dir_config, exist_ok=True)
    fn_config = f"{dir_config}/config{tag_inf}.yaml"
    if not overwrite and os.path.exists(fn_config):
        print(f"Config file already exists: {fn_config}")
        print("Set overwrite=True to overwrite the existing file.")
        return
    else:
        if os.path.exists(fn_config):
            print("Config file already exists but overwrite=True, overwriting. Hope you meant to do that!")
        with open(fn_config, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        print(f"Training config file written: {fn_config}")


def generate_test_config(dir_config=str(DEFAULT_CONFIGS_TEST_DIR),
                         overwrite=False,
                         statistics=['pk'],
                         n_train=N_TRAIN_SWEEP, bx=BX_SWEEP):
    """
    Generates a YAML configuration file for testing.
    """
    ### Select trained model
    #data_mode = 'emu'
    data_mode = 'muchisimocks'
    
    ### train params
    tag_params = '_p5_n10000'
    tag_biasparams = '_biasnest_p4_n320000'
    tag_noise = None
    # tag_biasparams = '_biasnoisenest_p9_n320000'
    # tag_noise = '_noise_unit_p5_n10000'
    #tag_mask = '_kb0.25'
    tag_mask = ''

    reparameterize = True
    tag_sweep = None  # set when loading best model from sweep (e.g. '-rand10')

    ##### test params
    data_mode_test = 'muchisimocks'
    idxs_obs = None # if none, all (unless evaluate mean)
    ### settings for fixed cosmo
    evaluate_mean = True
    tag_params_test = '_shame_p0_n1000'
    tag_biasparams_test = '_biasshame_p0_n1'
    tag_noise_test = None
    ### settings for coverage test
    #evaluate_mean = False
    #tag_params_test = '_coverage_p5_n1000'
    #tag_biasparams_test = '_biascoverage_p4_n1000'
    #tag_noise_test = None    # tag_biasparams_test = '_biasnoisecoverage_p9_n1000'
    # tag_noise_test = '_noise_unit_coverage_p5_n1000'

    
    # don't need train kwargs here bc not actually loading the data; just getting tag to reload model
    tag_stats = f'_{"_".join(statistics)}'    

    tags_mask = [tag_mask if stat == 'bispec' else '' for stat in statistics]
    tag_masks = ''.join(tags_mask)

    tag_paramsall = tag_params + tag_biasparams
    if tag_noise is not None:
        tag_paramsall += tag_noise
    tag_data_train = '_'+data_mode + tag_stats + tag_masks + tag_paramsall
    
    tag_paramsall_test = tag_params_test + tag_biasparams_test
    if tag_noise_test is not None:
        tag_paramsall_test += tag_noise_test
    tag_data_test = '_'+data_mode + tag_stats + tag_masks + tag_paramsall_test

    # build tag (run_mode fixed to load for testing)
    tag_inf_num = f'_bx{bx}_ntrain{n_train}'
    base_inf_train = tag_data_train + ('_rp' if reparameterize else '') + tag_inf_num
    tag_inf_num_sweep = f'_bx{BX_SWEEP}_ntrain{N_TRAIN_SWEEP}'
    base_inf_train_sweep = tag_data_train + ('_rp' if reparameterize else '') + tag_inf_num_sweep
    if tag_sweep is not None:
        tag_inf_train = base_inf_train + f'_best{tag_sweep}'
        sweep_name = base_inf_train_sweep + f'_sweep{tag_sweep}'
    else:
        tag_inf_train = base_inf_train
        sweep_name = None

    if evaluate_mean:
        tag_mean = '_mean'
    else:
        tag_mean = ''
    tag_test = f"_TRAIN{tag_inf_train}_TEST{tag_data_test}{tag_mean}"
    
    config = {
        "data_mode": data_mode,
        "data_mode_test": data_mode_test,
        "statistics": statistics,
        "tag_params": tag_params,
        "tag_biasparams": tag_biasparams,
        "tag_noise": tag_noise,
        "tags_mask": tags_mask,
        "tag_params_test": tag_params_test,
        "tag_biasparams_test": tag_biasparams_test,
        "tag_noise_test": tag_noise_test,
        "n_train": n_train,
        "bx": bx,
        "tag_sweep": tag_sweep,
        "evaluate_mean": evaluate_mean,
        "idxs_obs": idxs_obs,
        "tag_data_train": tag_data_train,
        "tag_inf_train": tag_inf_train,
        "sweep_name": sweep_name,
        "tag_data_test": tag_data_test,
        "tag_test": tag_test,
        "reparameterize": reparameterize,
    }
    
    os.makedirs(dir_config, exist_ok=True)
    fn_config = f"{dir_config}/config{tag_test}.yaml"
    if not overwrite and os.path.exists(fn_config):
        print(f"Config file already exists: {fn_config}")
        print("Set overwrite=True to overwrite the existing file.")
        return
    else:
        if os.path.exists(fn_config):
            print("Config file already exists but overwrite=True, overwriting. Hope you meant to do that!")
        with open(fn_config, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        print(f"Testing config file written: {fn_config}")
        
        
def generate_test_config_ood(dir_config=str(DEFAULT_CONFIGS_TEST_DIR),
                         overwrite=False,
                         statistics=['pk'],
                         n_train=N_TRAIN_SWEEP, bx=BX_SWEEP):
    """
    Generates a YAML configuration file for OOD testing.
    """
    ### Select trained model
    #data_mode = 'emu'
    data_mode = 'muchisimocks'
    
    ### train params
    tag_params = '_p5_n10000'
    #tag_biasparams = '_biasnoisenest_p9_n320000'
    #tag_noise = '_noise_unit_p5_n10000'
    tag_biasparams = '_biasnest_p4_n320000'
    tag_noise = None
    #tag_mask = '_kb0.25'
    tag_mask = ''

    reparameterize = True
    tag_sweep = None  # set when loading best model from sweep (e.g. '-rand10')

    ### test params
    idxs_obs = None # if none, all (unless evaluate mean)
    evaluate_mean = False
    data_mode_test = 'shame'
    tag_mock = '_nbar0.00022'
    #tag_mock = '_nbar0.00054' 
    #tag_mock = '_An1_orig_phase0' 
    
    ### train tags
    # don't need train kwargs here bc not actually loading the data; just getting tag to reload model
    tag_stats = f'_{"_".join(statistics)}'    
    tags_mask = [tag_mask if stat == 'bispec' else '' for stat in statistics]
    tag_masks = ''.join(tags_mask)
    
    tag_paramsall = tag_params + tag_biasparams
    if tag_noise is not None:
        tag_paramsall += tag_noise
    tag_data_train = '_'+data_mode + tag_stats + tag_masks + tag_paramsall

    # build tag (run_mode fixed to load for testing)
    tag_inf_num = f'_bx{bx}_ntrain{n_train}'
    base_inf_train = tag_data_train + ('_rp' if reparameterize else '') + tag_inf_num
    tag_inf_num_sweep = f'_bx{BX_SWEEP}_ntrain{N_TRAIN_SWEEP}'
    base_inf_train_sweep = tag_data_train + ('_rp' if reparameterize else '') + tag_inf_num_sweep
    if tag_sweep is not None:
        tag_inf_train = base_inf_train + f'_best{tag_sweep}'
        sweep_name = base_inf_train_sweep + f'_sweep{tag_sweep}'
    else:
        tag_inf_train = base_inf_train
        sweep_name = None

    ### test tags
    tag_data_test = '_'+data_mode_test + tag_stats + tag_masks + tag_mock
    
    if evaluate_mean:
        tag_mean = '_mean'
    else:
        tag_mean = ''
    tag_test = f"_TRAIN{tag_inf_train}_TEST{tag_data_test}{tag_mean}"
    
    config = {
        "data_mode": data_mode,
        "data_mode_test": data_mode_test,
        "statistics": statistics,
        "tag_params": tag_params,
        "tag_biasparams": tag_biasparams,
        "tag_noise": tag_noise,
        "tags_mask": tags_mask,
        "n_train": n_train,
        "bx": bx,
        "tag_sweep": tag_sweep,
        "evaluate_mean": evaluate_mean,
        "idxs_obs": idxs_obs,
        "tag_data_train": tag_data_train,
        "tag_inf_train": tag_inf_train,
        "sweep_name": sweep_name,
        "tag_data_test": tag_data_test,
        "tag_test": tag_test,
        "tag_mock": tag_mock,
        "reparameterize": reparameterize,
    }
    
    os.makedirs(dir_config, exist_ok=True)
    fn_config = f"{dir_config}/config{tag_test}.yaml"
    if not overwrite and os.path.exists(fn_config):
        print(f"Config file already exists: {fn_config}")
        print("Set overwrite=True to overwrite the existing file.")
        return
    else:
        if os.path.exists(fn_config):
            print("Config file already exists but overwrite=True, overwriting. Hope you meant to do that!")
        with open(fn_config, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        print(f"Testing config file written: {fn_config}")


def generate_runlike_config(dir_config=str(DEFAULT_CONFIGS_RUNLIKE_DIR), overwrite=False):
    """
    Generates a YAML configuration file for likelihood-based inference.
    """
    #data_mode = 'emu'  # or 'muchisimocksPk'
    data_mode = 'muchisimocks'
    statistics = ['pk'] 
    # i think i should make these "test" because no training, just evaluation!
    tag_params = '_quijote_p0_n1000'
    tag_biasparams = '_b1000_p0_n1'

    # Parameters to vary
    n_cosmo_params_vary = 5  # Number of cosmological parameters to vary
    n_bias_params_vary = 0  # Number of bias parameters to vary
    cosmo_param_names_vary = utils_model.cosmo_param_names_ordered[:n_cosmo_params_vary]
    bias_param_names_vary = utils_model.biasparam_names_ordered[:n_bias_params_vary]
    mcmc_framework = 'dynesty'  # or 'emcee'
    evaluate_mean = True
    #idxs_obs = [0]  # or None for all or evaluate_mean=True
    idxs_obs = None
    
    tag_stats = f'_{"_".join(statistics)}'    
    tag_data = '_'+data_mode + tag_stats + tag_params + tag_biasparams
    if evaluate_mean:
        tag_mean = '_mean'
    else:
        tag_mean = ''

    tag_inf = f'{tag_data}{tag_mean}_pvary{len(cosmo_param_names_vary)}_bvary{len(bias_param_names_vary)}'

    config = {
        'data_mode': data_mode,
        'statistics': statistics,
        'tag_params': tag_params,
        'tag_biasparams': tag_biasparams,
        'tag_data': tag_data,
        'tag_inf': tag_inf,
        'cosmo_param_names_vary': cosmo_param_names_vary,
        'bias_param_names_vary': bias_param_names_vary,
        'mcmc_framework': mcmc_framework,
        'evaluate_mean': evaluate_mean,
        'idxs_obs': idxs_obs,
    }

    os.makedirs(dir_config, exist_ok=True)
    fn_config = f"{dir_config}/config{tag_inf}.yaml"
    if not overwrite and os.path.exists(fn_config):
        print(f"Config file already exists: {fn_config}")
        print("Set overwrite=True to overwrite the existing file.")
        return
    else:
        if os.path.exists(fn_config):
            print("Config file already exists but overwrite=True, overwriting. Hope you meant to do that!")
        with open(fn_config, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        print(f"Runlike config file written: {fn_config}")


if __name__ == "__main__":
    main()
