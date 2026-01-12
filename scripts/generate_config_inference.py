import os
import yaml

import utils

'''
Generates a YAML configuration file for inference.
'''


def main():
    overwrite = False
    #overwrite = True
    #generate_train_config(overwrite=overwrite)
    #stat_arr = [['bispec'], ['pk', 'bispec']]
    stat_arr = [['pk']]
    #stat_arr = [['bispec']]
    n_train_arr = [10000]
    #stat_arr = [['pk'], ['bispec'], ['pk', 'bispec']]
    #n_train_arr = [500, 1000, 2000, 4000, 6000, 8000, 10000]
    for statistics in stat_arr:
        for n_train in n_train_arr:
            #generate_train_config(overwrite=overwrite, statistics=statistics, n_train=n_train)
            generate_test_config(overwrite=overwrite, statistics=statistics, n_train=n_train)
            #generate_test_config_ood(overwrite=overwrite, statistics=statistics, n_train=n_train)
    #generate_runlike_config(overwrite=overwrite)
    
    
def generate_train_config(dir_config='../configs/configs_train',
                          overwrite=False,
                          statistics=['pk'], n_train=10000):
    """
    Generates a YAML configuration file for training.
    """
    data_mode = 'muchisimocks'
    #data_mode = 'emu'
    #n_train = 1000
    #n_train = 10
    #n_train = None #if None, uses all (and no ntrain tag in tag_inf)
    tag_params = '_p5_n10000'
    #tag_biasparams = '_b1000_p0_n1'
    #tag_biasparams = '_b1zen_p1_n10000'
    #tag_biasparams = '_biaszen_p4_n10000' #1-1 cosmo-bias params
    #tag_biasparams = '_biaszen_p4_n50000' #5 bias params per cosmo
    #tag_biasparams = '_biaszen_p4_n100000' #10 bias params per cosmo
    tag_biasparams = '_biaszen_p4_n200000' #20 bias params per cosmo
    tag_noise = '_noise_unit_p5_n10000'
    #tag_Anoise = '_An_p1_n10000'
    #tag_Anoise = '_An1_p0_n1' #fix Anoise=1
    tag_Anoise = '_Anmult_p5_n200000'
    #tag_Anoise = '_Anmult_p5_n10000'
    tag_mask = '_kmaxbispec0.25'
    
    # emu-specific
    n_rlzs_per_cosmo = 1
    
    # running inferece params
    run_mode = 'single'
    tag_sweep = None
    n_train_sweep = None
    #run_mode = 'best'
    #run_mode = 'sweep'
    #tag_sweep = '-rand10'
    #n_train_sweep = 10000 # grab the hyperparams from the sweep trained on this many
        
    if data_mode == 'emu':
        tag_errG = '_boxsize1000'
        tag_datagen = f'{tag_errG}_nrlzs{n_rlzs_per_cosmo}'
        kwargs_data = {'n_rlzs_per_cosmo': n_rlzs_per_cosmo,
                    'tag_errG': tag_errG,
                    'tag_datagen': tag_datagen}
    elif data_mode == 'muchisimocks':
        tag_errG = None
        tag_datagen = ''
        kwargs_data = {'tag_datagen': tag_datagen}

    tag_stats = f'_{"_".join(statistics)}'    
    tag_paramsall = tag_params + tag_biasparams + tag_noise + tag_Anoise
    tag_data = '_'+data_mode + tag_stats + tag_mask + tag_paramsall + tag_datagen
    
    # build tag
    tag_inf = tag_data
    if n_train is not None:
        tag_inf += f'_ntrain{n_train}'
    if run_mode == 'sweep':
        tag_inf += f'_sweep{tag_sweep}'
        sweep_name = tag_inf
    elif run_mode == 'best':
        # if want best, neeed the sweep name to match sweep,
        # but new tag_inf will be best
        # sweep name is tag_inf of sweep; reconstruct cuz is diff than this tag_inf
        sweep_name = tag_data + f'_ntrain{n_train_sweep}_sweep{tag_sweep}'
        tag_inf += f'_best{tag_sweep}'
    elif run_mode == 'single':
        sweep_name = None
            
    config = {
        "data_mode": data_mode,
        "statistics": statistics,
        "tag_params": tag_params,
        "tag_biasparams": tag_biasparams,
        "tag_noise": tag_noise,
        "tag_Anoise": tag_Anoise,
        "tag_mask": tag_mask,
        "n_train": n_train,
        "kwargs_data": kwargs_data,
        "run_mode": run_mode,
        "sweep_name": sweep_name,
        "tag_data": tag_data,
        "tag_inf": tag_inf,
        "n_rlzs_per_cosmo": n_rlzs_per_cosmo,
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


def generate_test_config(dir_config='../configs/configs_test',
                         overwrite=False, 
                         statistics=['pk'], n_train=10000):
    """
    Generates a YAML configuration file for testing.
    """

    ### Select trained model
    #data_mode = 'emu'
    data_mode = 'muchisimocks'
    
    ### train params
    tag_params = '_p5_n10000'
    #tag_biasparams = '_b1000_p0_n1'
    #tag_biasparams = '_b1zen_p1_n10000'
    #tag_biasparams = '_biaszen_p4_n10000' #1x
    #tag_biasparams = '_biaszen_p4_n50000' #5x
    #tag_biasparams = '_biaszen_p4_n100000' #10x
    tag_biasparams = '_biaszen_p4_n200000' #20 bias params per cosmo
    tag_noise = '_noise_unit_p5_n10000'
    #tag_Anoise = '_An_p1_n10000'
    #tag_Anoise = '_An1_p0_n1' #fix Anoise=1
    #tag_Anoise = '_Anmult_p2_n10000'
    tag_Anoise = '_Anmult_p5_n200000'
    #tag_noise = None
    #tag_Anoise = None
    #tag_mask = '_kmaxbispec0.25'
    tag_mask = ''

    n_rlzs_per_cosmo = 1
    # For loading a model trained with wandb sweep; best of that sweep will be used
    #tag_sweep = '-rand10'
    #n_train_sweep = 10000
    tag_sweep = None
    n_train_sweep = None
    
    ### test params
    data_mode_test = 'muchisimocks'
    idxs_obs = None # if none, all (unless evaluate mean)
    ## settings for fixed cosmo
    # evaluate_mean = True
    # tag_params_test = '_quijote_p0_n1000'
    # tag_biasparams_test = '_b1000_p0_n1'
    # # tag_noise_test = None
    # # tag_Anoise_test = None
    # tag_noise_test = '_noise_unit_quijote_p0_n1000'
    # tag_Anoise_test = '_Anmult_p0_n1'
    ## settings for coverage test
    evaluate_mean = False
    tag_params_test = '_test_p5_n1000'
    tag_biasparams_test = '_biaszen_p4_n1000'
    tag_noise_test = '_noise_unit_test_p5_n1000'
    tag_Anoise_test = '_Anmult_p5_n1000'
    
    # this if-else is just so it's easier for me to switch between the two; may not need
    if data_mode_test == 'emu':
        # train
        tag_errG = '_boxsize1000'
        tag_datagen = f'{tag_errG}_nrlzs{n_rlzs_per_cosmo}'
        # test
        tag_errG_test = '_boxsize1000'
        tag_noiseless = ''
        #tag_noiseless = '_noiseless' # if use noiseless, set evaluate_mean=False (?)
        tag_datagen_test = f'{tag_errG_test}_nrlzs{n_rlzs_per_cosmo}'
        kwargs_data_test = {'n_rlzs_per_cosmo': n_rlzs_per_cosmo,
                            'tag_errG': tag_errG,
                            'tag_datagen': tag_datagen,
                            'tag_noiseless': tag_noiseless}
    elif data_mode_test == 'muchisimocks':
        # train
        tag_datagen = ''
        # test
        tag_noiseless = ''
        tag_datagen_test = ''
        kwargs_data_test = {
                            'tag_datagen': tag_datagen,
                            }
    
    # don't need train kwargs here bc not actually loading the data; just getting tag to reload model
    tag_stats = f'_{"_".join(statistics)}'    
    
    # NOTE for now just using one tag_mask for train & test bc i think they must be the same
    # but keep alert in case i need to change this later
    tag_paramsall = tag_params + tag_biasparams
    if tag_noise is not None:
        tag_paramsall += tag_noise + tag_Anoise
    tag_data_train = '_'+data_mode + tag_stats + tag_mask + tag_paramsall + tag_datagen
    
    tag_paramsall_test = tag_params_test + tag_biasparams_test
    if tag_noise_test is not None:
        tag_paramsall_test += tag_noise_test + tag_Anoise_test
    tag_data_test = '_'+data_mode + tag_stats + tag_mask + tag_paramsall_test + tag_datagen_test + tag_noiseless

    # build tag
    tag_inf_train = tag_data_train
    if n_train is not None:
        tag_inf_train += f'_ntrain{n_train}'
    # run_mode fixed to load for testing, bc always should be loading already trained model;
    # (best is for training; will read the best hyperparameters from a sweep and retrain and save it)
    if tag_sweep is not None:
        sweep_name = tag_data_train + f'_ntrain{n_train_sweep}_best{tag_sweep}'
    else:
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
        "tag_Anoise": tag_Anoise,
        "tag_mask": tag_mask,
        "tag_params_test": tag_params_test,
        "tag_biasparams_test": tag_biasparams_test,
        "tag_noise_test": tag_noise_test,
        "tag_Anoise_test": tag_Anoise_test,
        "kwargs_data_test": kwargs_data_test,
        "n_train": n_train,
        "evaluate_mean": evaluate_mean,
        "idxs_obs": idxs_obs,
        "tag_data_train": tag_data_train,
        "tag_inf_train": tag_inf_train,
        "sweep_name": sweep_name,
        "tag_data_test": tag_data_test,
        "tag_test": tag_test,
        "n_rlzs_per_cosmo": n_rlzs_per_cosmo,
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
        
        
def generate_test_config_ood(dir_config='../configs/configs_test',
                         overwrite=False, 
                         statistics=['pk'], n_train=10000):
    """
    Generates a YAML configuration file for testing.
    """

    ### Select trained model
    #data_mode = 'emu'
    data_mode = 'muchisimocks'
    
    ### train params
    tag_params = '_p5_n10000'
    #tag_biasparams = '_b1000_p0_n1'
    #tag_biasparams = '_b1zen_p1_n10000'
    #tag_biasparams = '_biaszen_p4_n10000' #1x
    #tag_biasparams = '_biaszen_p4_n50000' #5x
    #tag_biasparams = '_biaszen_p4_n100000' #10x
    tag_biasparams = '_biaszen_p4_n200000' #20 bias params per cosmo
    tag_noise = '_noise_unit_p5_n10000'
    tag_Anoise = '_Anmult_p5_n200000'
    #tag_Anoise = '_An_p1_n10000'
    #tag_Anoise = '_An1_p0_n1' #fix Anoise=1
    tag_datagen = ''
    #tag_noise = None
    #tag_Anoise = None
    tag_mask = '_kmaxbispec0.25'

    n_rlzs_per_cosmo = 1
    # For loading a model trained with wandb sweep; best of that sweep will be used
    #tag_sweep = '-rand10'
    #n_train_sweep = 10000
    tag_sweep = None
    n_train_sweep = None
    
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
    
    tag_paramsall = tag_params + tag_biasparams
    if tag_noise is not None:
        tag_paramsall += tag_noise + tag_Anoise
    tag_data_train = '_'+data_mode + tag_stats + tag_mask + tag_paramsall + tag_datagen

    # build tag
    tag_inf_train = tag_data_train
    if n_train is not None:
        tag_inf_train += f'_ntrain{n_train}'
    # run_mode fixed to load for testing, bc always should be loading already trained model;
    # (best is for training; will read the best hyperparameters from a sweep and retrain and save it)
    if tag_sweep is not None:
        sweep_name = tag_data_train + f'_ntrain{n_train_sweep}_best{tag_sweep}'
    else:
        sweep_name = None
    
    ### test tags
    # NOTE for now just using one tag_mask for train & test bc i think they must be the same
    # but keep alert in case i need to change this later
    tag_data_test = '_'+data_mode_test + tag_stats + tag_mask + tag_mock
    
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
        "tag_Anoise": tag_Anoise,
        "tag_mask": tag_mask,
        "n_train": n_train,
        "evaluate_mean": evaluate_mean,
        "idxs_obs": idxs_obs,
        "tag_data_train": tag_data_train,
        "tag_inf_train": tag_inf_train,
        "sweep_name": sweep_name,
        "tag_data_test": tag_data_test,
        "tag_test": tag_test,
        "tag_mock": tag_mock,
        "n_rlzs_per_cosmo": n_rlzs_per_cosmo,
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


def generate_runlike_config(dir_config='../configs/configs_runlike', overwrite=False):
    """
    Generates a YAML configuration file for likelihood-based inference.
    """
    #data_mode = 'emu'  # or 'muchisimocksPk'
    data_mode = 'muchisimocks'
    statistics = ['pk'] 
    # i think i should make these "test" because no training, just evaluation!
    tag_params = '_quijote_p0_n1000'
    tag_biasparams = '_b1000_p0_n1'
    n_rlzs_per_cosmo = 1

    # Parameters to vary
    n_cosmo_params_vary = 5  # Number of cosmological parameters to vary
    n_bias_params_vary = 0  # Number of bias parameters to vary
    cosmo_param_names_vary = utils.cosmo_param_names_ordered[:n_cosmo_params_vary]
    bias_param_names_vary = utils.biasparam_names_ordered[:n_bias_params_vary]
    mcmc_framework = 'dynesty'  # or 'emcee'
    evaluate_mean = True
    #idxs_obs = [0]  # or None for all or evaluate_mean=True
    idxs_obs = None

    # this if-else is just so it's easier for me to switch between the two; may not need
    if data_mode == 'emu':
        tag_errG = '_boxsize1000'
        tag_noiseless = ''
        #tag_noiseless = '_noiseless' # if use noiseless, set evaluate_mean=False (?)
        tag_datagen = f'{tag_errG}_nrlzs{n_rlzs_per_cosmo}'
        kwargs_data = {'n_rlzs_per_cosmo': n_rlzs_per_cosmo,
                            'tag_errG': tag_errG,
                            'tag_datagen': tag_datagen,
                            'tag_noiseless': tag_noiseless}
    elif data_mode == 'muchisimocks':
        # train
        tag_datagen = ''
        # test
        tag_noiseless = ''
        tag_datagen = ''
        kwargs_data = {
                            'tag_datagen': tag_datagen,
                            }
    
    tag_stats = f'_{"_".join(statistics)}'    
    tag_data = '_'+data_mode + tag_stats + tag_params + tag_biasparams + tag_datagen
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
        'n_rlzs_per_cosmo': n_rlzs_per_cosmo,
        'tag_datagen': tag_datagen,
        'tag_data': tag_data,
        'tag_inf': tag_inf,
        'cosmo_param_names_vary': cosmo_param_names_vary,
        'bias_param_names_vary': bias_param_names_vary,
        'mcmc_framework': mcmc_framework,
        'evaluate_mean': evaluate_mean,
        'idxs_obs': idxs_obs,
        'kwargs_data': kwargs_data
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
