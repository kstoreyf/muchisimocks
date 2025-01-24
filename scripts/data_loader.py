import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf

import utils



def load_data(data_mode, tag_params, tag_biasparams,
              kwargs={}):
    
    tag_mocks = tag_params + tag_biasparams
    if data_mode == 'emuPk':
        k, y, y_err = load_data_emuPk(tag_mocks, **kwargs)
    elif data_mode == 'muchisimocksPk':
        k, y, y_err = load_data_muchisimocksPk(tag_mocks, **kwargs)
    else:
        raise ValueError(f"Data mode {data_mode} not recognized!")
                   

    params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, random_ints, random_ints_bias = load_params(tag_params, tag_biasparams)
    return k, y, y_err, params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, random_ints, random_ints_bias


def load_data_muchisimocksPk(tag_mocks, tag_datagen=''):       
    
    # if 'fixedcosmo' in tag_mocks:
    #     param_dict = utils.cosmo_dict_quijote
    # else:
    #     dir_params = '../data/params'
    #     fn_params = f'{dir_params}/params_lh{tag_mocks}.txt'
    #     params_df = pd.read_csv(fn_params, index_col=0)
    
    # NOTE for now theta is only cosmo params! 
    # may want to add bias params too 
    
    #param_names = params_df.columns.tolist()
    #idxs_LH = params_df.index.tolist()

    dir_pks = f'/scratch/kstoreyf/muchisimocks/data/pks_mlib/pks{tag_mocks}{tag_datagen}'
    #idxs_LH = np.array([idx_LH for idx_LH in params_df.index.values 
    #                    if os.path.exists(f"{dir_pks}/pk_{idx_LH}.npy")])
    
    # using regexp bc not loading params_df here (tho could reorg if decide i need)
    idxs_LH = [int(re.search(r'pk_(\d+)\.npy', file_name).group(1)) 
               for file_name in os.listdir(dir_pks) if re.search(r'pk_(\d+)\.npy', file_name)]

    assert len(idxs_LH) > 0, f"No pks found in {dir_pks}!"

    #theta, Pk, gaussian_error_pk = [], [], []
    Pk, gaussian_error_pk = [], []
    for idx_LH in idxs_LH:
        fn_pk = f'{dir_pks}/pk_{idx_LH}.npy'
        pk_obj = np.load(fn_pk, allow_pickle=True).item()
        Pk.append(pk_obj['pk'])
        gaussian_error_pk.append(pk_obj['pk_gaussian_error'])
        # if 'fixedcosmo' in tag_mocks:
        #     theta.append(param_dict.values)
        # else:
        #     param_vals = params_df.loc[idx_LH].values
        #     theta.append(param_vals)

    Pk = np.array(Pk)
    #theta = np.array(theta)
    gaussian_error_pk = np.array(gaussian_error_pk)
    k = pk_obj['k'] # all ks should be same so just grab one
    
    return k, Pk, gaussian_error_pk


# hopefully this will become irrelevant after refactor !! 
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


def load_params(tag_params, tag_biasparams,
                dir_params='../data/params'):
        
    fn_params = f'{dir_params}/params_lh{tag_params}.txt'
    fn_params_fixed = f'{dir_params}/params_fixed{tag_params}.txt'
    
    params_df = (
        pd.read_csv(fn_params, index_col=0)
        if os.path.exists(fn_params)
        else None
    )
    param_dict_fixed = (
        pd.read_csv(fn_params_fixed).iloc[0].to_dict() 
        if os.path.exists(fn_params_fixed)
        else {}
    )
    #param_names = params_df.columns.tolist()
    # theta = params_df.values
    # assert theta.shape[0] == n_data, f"Expected {n_data} rows in {fn_params}"
    
    fn_biasparams = f'{dir_params}/params_lh{tag_biasparams}.txt'
    fn_biasparams_fixed = f'{dir_params}/params_fixed{tag_biasparams}.txt'
    biasparams_df = (
        pd.read_csv(fn_biasparams, index_col=0)
        if os.path.exists(fn_biasparams)
        else None
    )
    biasparams_dict_fixed = (
        pd.read_csv(fn_biasparams_fixed).iloc[0].to_dict() 
        if os.path.exists(fn_biasparams_fixed)
        else {}
    )
    
    fn_randints = f'{dir_params}/randints{tag_params}.npy'
    fn_randints_bias = f'{dir_params}/randints{tag_biasparams}.npy'
    random_ints = np.load(fn_randints, allow_pickle=True) if os.path.exists(fn_randints) else None
    random_ints_bias = np.load(fn_randints_bias, allow_pickle=True) if os.path.exists(fn_randints_bias) else None
    
    return params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, random_ints, random_ints_bias
    
    
def param_dfs_to_theta(params_df, biasparams_df, n_rlzs_per_cosmo=1):
    assert params_df is not None or biasparams_df is not None, "params_df or biasparams_df (or both) must be specified"
    param_names = []
    if params_df is not None:
        param_names.extend(params_df.columns.tolist())
        theta_cosmo_orig = params_df.values
        theta_cosmo = utils.repeat_arr_rlzs(theta_cosmo_orig, n_rlzs=n_rlzs_per_cosmo)
    if biasparams_df is not None:
        param_names.extend(biasparams_df.columns.tolist())
        theta_bias_orig = biasparams_df.values
        theta_bias = utils.repeat_arr_rlzs(theta_bias_orig, n_rlzs=n_rlzs_per_cosmo)

    if params_df is not None and biasparams_df is not None:
        theta = np.concatenate((theta_cosmo, theta_bias), axis=1)
    else:
        theta = theta_cosmo if params_df is not None else theta_bias

    return theta, param_names
    

def load_data_emuPk(tag_mocks, tag_errG='', tag_datagen='', tag_noiseless='',
                    n_rlzs_per_cosmo=1, tag_mask=''):
    
    assert tag_errG is not None, "tag_errG must be specified"
    dir_emuPk = f'../data/emuPks/emuPks{tag_mocks}'
    
    assert tag_noiseless in ['', '_noiseless'], "tag_noiseless must be '_noiseless' or ''"
    if 'noiseless' in tag_noiseless:
        assert n_rlzs_per_cosmo==1, "Why would you want multiple realizations per cosmo if using noiseless?"
        fn_emuPk = f'{dir_emuPk}/emuPks.npy'
    else:
        fn_emuPk = f'{dir_emuPk}/emuPks_noisy{tag_datagen}.npy'
    fn_emuk = f'{dir_emuPk}/emuPks_k.txt'
    fn_emuPkerrG = f'{dir_emuPk}/emuPks_errgaussian{tag_errG}.npy'
    #fn_emuPk_params = f'{dir_emuPk}/params_lh{tag_params}.txt'
    #fn_bias_vector = f'{dir_emuPk}/bias_params.txt'

    Pk = np.load(fn_emuPk, allow_pickle=True)   
    k = np.genfromtxt(fn_emuk)

    #params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed = load_params(tag_params, tag_biasparams)
    #theta_noiseless = np.genfromtxt(fn_emuPk_params, delimiter=',', names=True)
    #print("theta_noiseless", theta_noiseless.shape)
    #param_names = theta_noiseless.dtype.names
    #theta_noiseless = np.array([list(tup) for tup in theta_noiseless]) # from tuples to 2d array
    
    # if we have more than 1 rlz per cosmo, repeat the error arrays to get the right number
    #theta = utils.repeat_arr_rlzs(theta_noiseless, n_rlzs=n_rlzs_per_cosmo)
    gaussian_error_pk_orig = np.load(fn_emuPkerrG, allow_pickle=True)
    gaussian_error_pk = utils.repeat_arr_rlzs(gaussian_error_pk_orig, n_rlzs=n_rlzs_per_cosmo)
    assert gaussian_error_pk.shape[0] == Pk.shape[0], "Number of pks and errors should be the same, something is wrong"
    
    # mask = np.all(Pk>0, axis=0)
    # print("mask")
    # print(mask)
    # Pk = Pk[:,mask]
    # gaussian_error_pk = gaussian_error_pk[:,mask]
    # k = k[mask]
    # print(len(k))

    return k, Pk, gaussian_error_pk


# used to both remove nonpositive data, and to select certain k_bins
def get_Pk_mask(tag_data, tag_mask='', k=None, Pk=None):
    dir_masks = '../data/masks'
    fn_mask = f'{dir_masks}/mask{tag_data}{tag_mask}.txt'
    if os.path.exists(fn_mask):
        return np.loadtxt(fn_mask, dtype=bool)
    else:
        if Pk is not None:
            mask = np.all(Pk>0, axis=0)
        else:
            mask = np.ones(len(k), dtype=int)
        np.savetxt(fn_mask, mask.astype(int), fmt='%i')
    return mask



# NOT USED/WORKING RN
def load_data_muchisimocks3D(tag_mocks, n_train, n_val, n_test,
                             batch_size=32, n_threads=8):       
    
    if n_threads is not None:
        tf.config.threading.set_inter_op_parallelism_threads(n_threads)
        tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    
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

    fns_fields, theta = [], []
    for idx_LH in idxs_LH:
        fns_fields.append( f'{dir_mocks}/LH{idx_LH}/bias_fields_eul{tag_fields}_{idx_LH}{tag_fields_extra}.npy' )
        # TODO what is my error??
        param_vals = params_df.loc[idx_LH].values
        theta.append(param_vals)

    fns_fields = np.array(fns_fields)
    theta = np.array(theta)
    #y_err = np.array(y_err)
        
    frac_train=0.8
    frac_val=0.1
    frac_test=0.1
        
    idxs_train, idxs_val, idxs_test = utils.idxs_train_val_test(random_ints, 
                                   frac_train=frac_train, frac_val=frac_val, frac_test=frac_test)
    idxs_train = idxs_train[:n_train]
    idxs_val = idxs_val[:n_val]
    idxs_test = idxs_test[:n_test]

    theta_train, theta_val, theta_test = utils.split_train_val_test(theta, idxs_train, idxs_val, idxs_test)
    fns_train, fns_val, fns_test = utils.split_train_val_test(fns_fields, idxs_train, idxs_val, idxs_test)
    #y_err_train, y_err_val, y_err_test = utils.split_train_val_test(y_err, idxs_train, idxs_val, idxs_test)

    print("load_data_muchisimocks3D")
    print(len(fns_train), len(fns_val), len(fns_test))
    print(idxs_train[:10], idxs_val[:10], idxs_test[:10])

    ds_fns_train = tf.data.Dataset.from_tensor_slices(fns_train)
    ds_theta_train = tf.data.Dataset.from_tensor_slices(theta_train)
    dataset_train = tf.data.Dataset.zip((ds_fns_train, ds_theta_train))
    
    ds_fns_val = tf.data.Dataset.from_tensor_slices(fns_val)
    ds_theta_val = tf.data.Dataset.from_tensor_slices(theta_val)
    dataset_val = tf.data.Dataset.zip((ds_fns_val, ds_theta_val))
    
    ds_fns_test = tf.data.Dataset.from_tensor_slices(fns_test)
    ds_theta_test = tf.data.Dataset.from_tensor_slices(theta_test)
    dataset_test = tf.data.Dataset.zip((ds_fns_test, ds_theta_test))
        
    # Step 3: Process the dataset

    def process_fn(fn_fields, theta_val):
        print('processing', fn_fields)
        bias_terms_eul = np.load(fn_fields.numpy().decode("utf-8"))
        print("loaded")
        tracer_field = utils.get_tracer_field(bias_terms_eul, bias_vector, n_grid_norm=n_grid_orig)
        print(np.min(tracer_field), np.max(tracer_field))
        tracer_field = tf.convert_to_tensor(tracer_field, dtype=tf.float32)
        #theta_val = tf.convert_to_tensor(np.float32(theta_val), dtype=tf.float32)
        theta_val = tf.convert_to_tensor(theta_val, dtype=tf.float64)
        theta_val = tf.cast(theta_val, tf.float32)
        print(tracer_field.shape, theta_val.shape)
        return tracer_field, theta_val

    # Wrapper function for use with tf.py_function
    def process_fn_py(fn_fields, theta_val):
        tracer_field, theta_val = tf.py_function(func=process_fn, inp=[fn_fields, theta_val], Tout=(tf.float32, tf.float32))
        #image.set_shape([224, 224, 3])  # Set the shape explicitly to [224, 224, 3]
        return tracer_field, theta_val

    # Apply the processing function to the dataset
    # dataset_train = dataset_train.map(process_fn_py, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset_val = dataset_val.map(process_fn_py, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset_test = dataset_test.map(process_fn_py, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print("Set up processing")
    dataset_train = dataset_train.map(lambda fn_fields, label: tf.py_function(
            process_fn_py, [fn_fields, label], [tf.float32, tf.float32]), 
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_val = dataset_val.map(lambda fn_fields, label: tf.py_function(
            process_fn_py, [fn_fields, label], [tf.float32, tf.float32]), 
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_test = dataset_test.map(lambda fn_fields, label: tf.py_function(
            process_fn_py, [fn_fields, label], [tf.float32, tf.float32]), 
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Step 4: Configure dataset for performance
    # print("Shuffle")
    # dataset_train = dataset_train.shuffle(buffer_size=len(fns_train)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    # dataset_val = dataset_val.shuffle(buffer_size=len(fns_val)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    # dataset_test = dataset_test.shuffle(buffer_size=len(fns_test)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset_train, dataset_val, dataset_test, param_names



# NOT CURRENTLY USED
def get_split(data_mode, split, theta, y, y_err, random_ints, 
              n_train, n_val, n_test,
              n_rlzs_per_cosmo=None):
    
    # keep these fractions hard-coded or else things will get weird
    frac_train=0.8
    frac_val=0.1
    frac_test=0.1

    print(len(random_ints))
    #TODO deal with N_tot hardcoding magic
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

    