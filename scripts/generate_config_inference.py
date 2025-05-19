import os
import yaml


def main():
    overwrite = False
    #overwrite = True
    generate_train_config(overwrite=overwrite)
    generate_test_config(overwrite=overwrite)
    
    
def generate_train_config(dir_config='../configs/configs_train',
                          overwrite=False):
    """
    Generates a YAML configuration file for training.
    """
    data_mode = 'muchisimocks'
    statistics = ['bispec']
    n_train = 1000
    #n_train = 10
    #n_train = None #if None, uses all (and no ntrain tag in tag_inf)
    tag_params = '_p5_n10000' #for emu, formerly tag_emuPk
    #tag_biasparams = '_b1000_p0_n1'
    #tag_biasparams = '_b1zen_p1_n10000'
    tag_biasparams = '_biaszen_p4_n10000'
    #tag_biasparams = '_biaszen_p4_n100000' #10 bias params per cosmo
    # emu-specific
    n_rlzs_per_cosmo = 1
    tag_errG = None
    
    # running inferece params
    run_mode = 'best'
    sweep_name = 'sbi-rand10'
    # tag_sweep = '_rand10'
    # sweep_name = tag_inf + tag_sweep
        
    if data_mode == 'emuPk':
        tag_datagen = f'{tag_errG}_nrlzs{n_rlzs_per_cosmo}'
        kwargs_data = {'n_rlzs_per_cosmo': n_rlzs_per_cosmo,
                    'tag_errG': tag_errG,
                    'tag_datagen': tag_datagen}
    elif data_mode == 'muchisimocks':
        tag_datagen = ''
        kwargs_data = {'tag_datagen': tag_datagen}

    tag_stats = f'_{"_".join(statistics)}'    
    tag_data = '_'+data_mode + tag_stats + tag_params + tag_biasparams + tag_datagen
    
    # build tag
    tag_inf = tag_data
    if n_train is not None:
        tag_inf += f'_ntrain{n_train}'
    if run_mode == 'sweep':
        tag_inf += f'_sweep-{sweep_name}'
    elif run_mode == 'best':
        tag_inf += f'_best-{sweep_name}'
            
    config = {
        "data_mode": data_mode,
        "statistics": statistics,
        "tag_params": tag_params,
        "tag_biasparams": tag_biasparams,
        "n_train": n_train,
        "evaluate_mean": False,
        "kwargs_data": kwargs_data,
        "run_mode": run_mode,
        "sweep_name": sweep_name,
        "tag_data": tag_data,
        "tag_inf": tag_inf,
        "n_rlzs_per_cosmo": n_rlzs_per_cosmo,
    }
            
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
                         overwrite=False):
    """
    Generates a YAML configuration file for testing.
    """

    ### Select trained model
    #data_mode = 'emuPk'
    data_mode = 'muchisimocks'
    statistics = ['bispec']
    
    ### train params
    tag_params = '_p5_n10000'
    #tag_biasparams = '_b1000_p0_n1'
    #tag_biasparams = '_b1zen_p1_n10000'
    tag_biasparams = '_biaszen_p4_n10000'
    #tag_biasparams = '_biaszen_p4_n100000'
    n_rlzs_per_cosmo = 1
    n_train = 1000
    # For loading a model trained with wandb sweep; best of that sweep will be used
    #sweep_name = None
    sweep_name = 'sbi-rand10'
    
    ### test params
    idxs_obs = None # if none, all (unless evaluate mean)
    ## settings for fixed cosmo
    #evaluate_mean = True 
    #tag_params_test = '_quijote_p0_n1000'
    #tag_biasparams_test = '_b1000_p0_n1'
    ## settings for coverage test
    evaluate_mean = False
    tag_params_test = '_test_p5_n1000'
    #tag_biasparams_test = '_b1000_p0_n1'
    #tag_biasparams_test = '_b1zen_p1_n1000'
    tag_biasparams_test = '_biaszen_p4_n1000'

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
    elif data_mode == 'muchisimocks':
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
    tag_data_train = '_'+data_mode + tag_stats + tag_params + tag_biasparams + tag_datagen
    tag_data_test = '_'+data_mode + tag_stats + tag_params_test + tag_biasparams_test + tag_datagen_test + tag_noiseless
    
    # build tag
    tag_inf_train = tag_data_train
    if n_train is not None:
        tag_inf_train += f'_ntrain{n_train}'
    if sweep_name is not None:
        # run_mode fixed to load for testing, bc always should be
        # loading already trained model
        # (best is for reading the best hyperparameters from a sweep and retraining and saving it)
        tag_inf_train += f'_best-{sweep_name}'
    
    if evaluate_mean:
        tag_mean = '_mean'
    else:
        tag_mean = ''
    tag_test = f"_TRAIN{tag_inf_train}_TEST{tag_data_test}{tag_mean}"
    
    config = {
        "data_mode": data_mode,
        "statistics": statistics,
        "tag_params": tag_params,
        "tag_biasparams": tag_biasparams,
        "tag_params_test": tag_params_test,
        "tag_biasparams_test": tag_biasparams_test,
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


if __name__ == "__main__":
    main()
