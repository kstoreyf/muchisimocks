import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf

import utils



def load_data(data_mode, statistics, tag_params, tag_biasparams,
              tag_data=None,
              kwargs={}):
    
    k, y, y_err = [], [], []
    for statistic in statistics:
        # idxs_params should be the same for all so will just grab last one
        if data_mode == 'muchisimocks':
            k_i, y_i, y_err_i, idxs_params = load_data_muchisimocks(statistic,
                                            tag_params, tag_biasparams, **kwargs)
        elif data_mode == 'emu':
            k_i, y_i, y_err_i, idxs_params = load_data_emu(statistic,
                                            tag_params, tag_biasparams, **kwargs)
        else:
            raise ValueError(f"Data mode {data_mode} not recognized!")
        print(f"Loaded {statistic} data with shape {y_i.shape}")
        if tag_data is None:
            print("No tag_data provided, so not masking data")
        else:
            k_i, y_i, y_err_i = mask_data(statistic, tag_data, k_i, y_i, y_err_i)

        k.append(k_i)
        y.append(y_i)
        y_err.append(y_err_i)    

    params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, random_ints, random_ints_bias = load_params(tag_params, tag_biasparams)
    return k, y, y_err, idxs_params, params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, random_ints, random_ints_bias


def get_idxs_params(tag_params, tag_biasparams, n_bias_per_cosmo=1, 
                    modecosmo='lh'):
    """
    For each cosmological parameter index, get the corresponding bias parameter indices.
    Returns a list of (idx_mock, idx_bias) tuples.
    """
    params_df, param_dict_fixed = load_cosmo_params(tag_params)
    biasparams_df, biasparams_dict_fixed = load_bias_params(tag_biasparams)
    n_cosmo_params = len(params_df)
    n_biasparams = len(biasparams_df)
    factor = n_biasparams // n_cosmo_params
    if n_biasparams % n_cosmo_params != 0:
        raise ValueError(f"biasparams_df length ({n_biasparams}) is not an integer multiple of params_df length ({n_cosmo_params})")

    idxs_params = []
    for idx_mock in params_df.index:
        if modecosmo == 'lh':
            start_idx = idx_mock * factor
            end_idx = (idx_mock + 1) * factor
            for idx_bias in range(start_idx, end_idx):
                idxs_params.append((idx_mock, idx_bias))
        elif modecosmo == 'fisher':
            if params_df.iloc[idx_mock]['param_shifted'] == 'fiducial':
                for idx_bias in biasparams_df.index:
                    idxs_params.append((idx_mock, idx_bias))
            else:
                fiducial_bias_idxs = np.where(biasparams_df['param_shifted'] == 'fiducial')[0]
                for idx_bias in fiducial_bias_idxs:
                    idxs_params.append((idx_mock, idx_bias))
        else:
            raise ValueError(f"Unknown modecosmo: {modecosmo}")
    return idxs_params


def get_bias_indices_for_idx(idx_mock, modecosmo='lh', factor=None,
                             params_df=None, biasparams_df=None):
    """
    # TODO this should probs be combined with check_df_lengths function,
    # but would have to revamp the params-to-theta function
    Get the bias parameter indices corresponding to a power spectrum index.
    
    For LH:
    For a given idx_LH and factor, returns consecutive indices:
    idx_LH=0, factor=10 -> [0, 1, 2, ..., 9]
    idx_LH=1, factor=10 -> [10, 11, 12, ..., 19]
    For Fisher:
    Get so only one param is shifted each time
    
    Args:
        idx_LH (int): The power spectrum index
        factor (int): The factor relating number of bias parameters to number of power spectra
        
    Returns:
        list: List of bias parameter indices
    """
    
    if modecosmo=='lh':
        assert factor is not None, "factor must be provided"
        start_idx = idx_mock * factor
        end_idx = (idx_mock + 1) * factor
        idxs_bias = list(range(start_idx, end_idx))
    elif modecosmo=='fisher':
        assert params_df is not None and biasparams_df is not None, "params_df and biasparams_df must be provided"
        if params_df.iloc[idx_mock]['param_shifted']=='fiducial':
            idxs_bias = biasparams_df.index
        else:
            idxs_bias = np.where(biasparams_df['param_shifted']=='fiducial')[0]        
    return idxs_bias



def check_df_lengths(params_df, biasparams_df):
    """
    Check the length relationship between params_df and biasparams_df.
    
    Args:
        params_df (pd.DataFrame): DataFrame containing cosmological parameters
        biasparams_df (pd.DataFrame): DataFrame containing bias parameters
        
    Returns:
        tuple: (factor, longer_df)
            - factor: The integer factor relating the lengths (1 if same length)
            - longer_df: String indicating which DataFrame is longer ('params', 'bias', or 'same')
    """
    if params_df is None or biasparams_df is None:
        return 1, 'same'
    
    n_params = len(params_df)
    n_biasparams = len(biasparams_df)
    
    if n_params == n_biasparams:
        return 1, 'same'
    
    if n_params > n_biasparams:
        factor = n_params / n_biasparams
        if not factor.is_integer():
            raise ValueError(f"params_df length ({n_params}) is not an integer multiple of biasparams_df length ({n_biasparams})")
        return int(factor), 'params'
    else:
        factor = n_biasparams / n_params
        if not factor.is_integer():
            raise ValueError(f"biasparams_df length ({n_biasparams}) is not an integer multiple of params_df length ({n_params})")
        return int(factor), 'bias'


def load_data_muchisimocks(statistic, tag_params, tag_biasparams, tag_datagen='',
                           mode_precomputed=None, return_pk_objs=False):
    
    print("Loading muchisimocks data")
    tag_mocks = tag_params + tag_biasparams
    
    dir_statistics = f'/scratch/kstoreyf/muchisimocks/data/{statistic}s_mlib/{statistic}s{tag_mocks}{tag_datagen}'
    
    # this will differ if not precomputed
    stat_name = statistic
    if os.path.exists(dir_statistics):
        # If the precomputed directory exists, use it
        mode_precomputed = True
    else:
        if statistic == 'pk':
            stat_name = 'pnn'
        dir_statistics = f'/scratch/kstoreyf/muchisimocks/data/{stat_name}s_mlib/{stat_name}s{tag_params}{tag_datagen}'
        mode_precomputed = False
    
    # Load bias parameters regardless of mode
    params_df, param_dict_fixed = load_cosmo_params(tag_params)
    biasparams_df, biasparams_dict_fixed = load_bias_params(tag_biasparams)
    
    #n_cosmos = int(re.search(r'n(\d+)', tag_params).group(1))
    #idxs_LH = np.arange(n_cosmos)
    # no ideal way to get this number...
    #idxs_LH = np.array(params_df.index)
    # using regexp bc not loading params_df here (tho could reorg if decide i need)
    idxs_LH = [int(re.search(fr'{stat_name}_(\d+)[^/]*\.npy', file_name).group(1))
                     for file_name in os.listdir(dir_statistics) if re.search(fr'{stat_name}_(\d+)[^/]*\.npy', file_name)]
    # need set bc for cases w multiple bias parameters per, want to only grab the cosmo
    idxs_LH = np.sort(np.array(list(set(idxs_LH)))) # now doing regexp, need to sort
    #idxs_LH = np.array([idx_LH for idx_LH in params_df.index.values
    #                    if os.path.exists(f"{dir_statistics}/pk_{idx_LH}.npy")])
    print(f"Found {len(idxs_LH)} diff cosmo {stat_name}s in {dir_statistics}")
    assert len(idxs_LH) > 0, f"No pks found in {dir_statistics}!"
    
    # Check the relationship between biasparams_df and idxs_LH
    if 'fisher' not in tag_biasparams:
        # only used later if not fisher
        factor, longer_df = check_df_lengths(params_df, biasparams_df)

    #theta, Pk, gaussian_error_pk = [], [], []
    stat_arr, error_arr, idxs_params = [], [], []
    if return_pk_objs:
        pk_obj_arr = []
    for i, idx_LH in enumerate(idxs_LH):
        #if idx_LH%100 == 0:
            #print(f"Loading Pk {idx_LH}", flush=True)
        
        # figure out which bias indices to use
        if 'fisher' in tag_biasparams:
            idxs_bias = get_bias_indices_for_idx(idx_LH, modecosmo='fisher', 
                                                params_df=params_df, biasparams_df=biasparams_df)
        else:
            assert longer_df == 'bias' or longer_df == 'same', "In non-precomputed mode, biasparams_df should be longer or same length as params_df"
            idxs_bias = get_bias_indices_for_idx(idx_LH, modecosmo='lh', factor=factor)
        
        if mode_precomputed:
            if statistic == 'pk':
                # TODO pk is old style where only 1-1 cosmo-bias, so this name format
                fn_stat = f'{dir_statistics}/{stat_name}_{idx_LH}.npy'
                pk_obj = np.load(fn_stat, allow_pickle=True).item()
                k = pk_obj['k'] # all ks should be same so just grab one
                stat, error = pk_obj['pk'], pk_obj['pk_gaussian_error']
                stat_arr.append(stat)
                error_arr.append(error)
                # if just one bias param per idx_cosmo, then just use the same
                idxs_params.append((idx_LH, idx_LH))
                if return_pk_objs:
                    pk_obj_arr.append(pk_obj)
                    
            elif statistic == 'bispec':
                for idx_bias in idxs_bias:
                    if biasparams_df is None:
                        # if fixed bias params, it wont be in the name
                        fn_stat = f'{dir_statistics}/{stat_name}_{idx_LH}.npy'
                    else:
                        fn_stat = f'{dir_statistics}/{stat_name}_{idx_LH}_b{idx_bias}.npy'
                    if not os.path.exists(fn_stat):
                        print(f"WARNING: Missing {fn_stat}, skipping")
                        continue
                    bispec_obj = np.load(fn_stat, allow_pickle=True).item()
                    # multiply by the weight and normalize to get a 
                    # better behaved stat
                    n_grid = 128
                    norm = n_grid**3
                    k = bispec_obj['k123']
                    stat = norm**3 * bispec_obj['weight']*bispec_obj['bispectrum']['b0']
                    stat_arr.append(stat)
                    idxs_params.append((idx_LH, idx_bias))
                    if return_pk_objs:
                        pk_obj_arr.append(bispec_obj)
        else:
            
            if statistic == 'pk':
                # here we havent selected bias params yet
                fn_stat = f'{dir_statistics}/{stat_name}_{idx_LH}.npy'
                pnn_obj = np.load(fn_stat, allow_pickle=True)
                k = pnn_obj[0]['k']
                if return_pk_objs:
                    pk_obj_arr.append(pnn_obj)
            else:
                raise ValueError(f"Statistic {statistic} not recognized for non-precomputed mode!")    
            
            for idx_bias in idxs_bias:
                if biasparams_df is not None:
                    bias_params_dict = biasparams_df.loc[idx_bias].to_dict()
                else:
                    # case where only fixed bias params
                    bias_params_dict = {}
                bias_params_dict.update(biasparams_dict_fixed)
                bias_params = [bias_params_dict[bpn] for bpn in utils.biasparam_names_ordered]
                
                if statistic == 'pk':
                    stat = utils.pnn_to_pk(pnn_obj, bias_params)
                stat_arr.append(stat)
                idxs_params.append((idx_LH, idx_bias))

    stat_arr = np.array(stat_arr)
    error_arr = np.array(error_arr)
    idxs_params= np.array(idxs_params)
    
    if len(error_arr)==0:
        error_arr = np.zeros_like(stat_arr)
        
    if return_pk_objs:
        return k, stat_arr, error_arr, idxs_params, pk_obj_arr
    else:
        return k, stat_arr, error_arr, idxs_params


def mask_data(statistic, tag_data, k, y, y_err, tag_mask=''):
    if statistic == 'pk':
        mask = get_Pk_mask(tag_data, tag_mask=tag_mask, k=k, Pk=y)
    else:
        print("No mask for this statistic, using all data")
        mask = [True]*y.shape[-1]
    print(f"Masked {np.sum(~np.array(mask))} out of {len(mask)} bins")
    if k.ndim == 1:
        k_masked = k[mask]
    elif k.ndim == 2:
        k_masked = k[:,mask]
    else:
        raise ValueError(f"Unexpected k shape: {k.shape}")
    return k_masked, y[:,mask], y_err[:,mask]


def load_params(tag_params=None, tag_biasparams=None,
                dir_params='../data/params'):
    
    if tag_params is None:
        params_df = None
        param_dict_fixed = {}
        random_ints_bias = None
    else:        
        params_df, param_dict_fixed = load_cosmo_params(tag_params, dir_params=dir_params)
        fn_randints = f'{dir_params}/randints{tag_params}.npy'
        random_ints = np.load(fn_randints, allow_pickle=True) if os.path.exists(fn_randints) else None

    if tag_biasparams is None:
        biasparams_df = None
        biasparams_dict_fixed = {}
        random_ints_bias = None
    else:
        biasparams_df, biasparams_dict_fixed = load_bias_params(tag_biasparams, dir_params=dir_params)
        fn_randints_bias = f'{dir_params}/randints{tag_biasparams}.npy'
        random_ints_bias = np.load(fn_randints_bias, allow_pickle=True) if os.path.exists(fn_randints_bias) else None
        
    return params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, random_ints, random_ints_bias
    
    
def load_cosmo_params(tag_params, dir_params='../data/params'):
    # TODO this is so messy! figure out what to do about lh prefix
    if 'fisher' in tag_params:
        fn_params = f'{dir_params}/params{tag_params}.txt'
    else:
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
    return params_df, param_dict_fixed
    
def load_bias_params(tag_biasparams, dir_params='../data/params'):
    if 'fisher' in tag_biasparams:
        fn_biasparams = f'{dir_params}/params{tag_biasparams}.txt'
    else:
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
    return biasparams_df, biasparams_dict_fixed
    
    
def param_dfs_to_theta(idxs_params, params_df, biasparams_df, n_rlzs_per_cosmo=1):
    """
    Convert parameter DataFrames to a theta array that can be used for inference.
    Handles cases where params_df and biasparams_df have different lengths.
    
    Args:
        params_df (pd.DataFrame): DataFrame containing cosmological parameters
        biasparams_df (pd.DataFrame): DataFrame containing bias parameters
        n_rlzs_per_cosmo (int): Number of realizations per cosmology
        
    Returns:
        tuple: (theta, param_names)
            - theta: 2D numpy array of parameter values
            - param_names: List of parameter names corresponding to columns in theta
    """
    assert params_df is not None or biasparams_df is not None, "params_df or biasparams_df (or both) must be specified"
    param_names = []

    # Base case: only one DataFrame provided
    if params_df is None:
        param_names.extend(biasparams_df.columns.tolist())
        theta_bias_orig = np.array([biasparams_df.loc[idx_bias].values for _, idx_bias in idxs_params])
        theta_bias = utils.repeat_arr_rlzs(theta_bias_orig, n_rlzs=n_rlzs_per_cosmo)
        return theta_bias, param_names
    elif biasparams_df is None:
        param_names.extend(params_df.columns.tolist())
        theta_cosmo_orig = np.array([params_df.loc[idx_mock].values for idx_mock, _ in idxs_params])
        theta_cosmo = utils.repeat_arr_rlzs(theta_cosmo_orig, n_rlzs=n_rlzs_per_cosmo)
        return theta_cosmo, param_names
    
    # Add parameter names from both DataFrames
    param_names.extend(params_df.columns.tolist())
    param_names.extend(biasparams_df.columns.tolist())

    theta = []
    for idx_mock, idx_bias in idxs_params:
        cosmo_params = params_df.loc[idx_mock].values
        bias_params = biasparams_df.loc[idx_bias].values
        params_combined = np.concatenate([cosmo_params, bias_params])
        theta.append(params_combined)        
    theta = np.array(theta)
    
    # TODO test/check this part! not using bc haven't been using emulated set
    # Apply realization repetition if needed
    if n_rlzs_per_cosmo > 1:
        theta = utils.repeat_arr_rlzs(theta, n_rlzs=n_rlzs_per_cosmo)
    
    return theta, param_names
    
    
    
def load_theta_test(tag_params_test, tag_biasparams_test,
                    cosmo_param_names_vary=None, bias_param_names_vary=None, n_rlzs_per_cosmo=1):
    # in theory could make this load whichever of the params want, variable or fixed,
    # but for now we want either completely fixed or completely varied, so this is fine
    
    # get parameter files
    params_df_test, param_dict_fixed_test, biasparams_df_test, biasparams_dict_fixed_test, _, _ = load_params(tag_params_test, tag_biasparams_test)
    
    # load test data
    if 'p0' in tag_params_test and 'p0' in tag_biasparams_test:
        msg = "If all fixed parameters in test set, need to specify which parameters to provide"
        assert cosmo_param_names_vary is not None and bias_param_names_vary is not None, msg
        # if both tests sets are entirely fixed, our theta is just the fixed data, repeated for each observation
        theta_test = [param_dict_fixed_test[pname] for pname in cosmo_param_names_vary]
        theta_test.extend([biasparams_dict_fixed_test[pname] for pname in bias_param_names_vary])
    else:
        # for when have a LH of varied bias parameters
        # TODO update to handle idxs_params!
        idxs_params = get_idxs_params(tag_params_test, tag_biasparams_test)
        theta_test, param_names = param_dfs_to_theta(idxs_params, params_df_test, biasparams_df_test, n_rlzs_per_cosmo=n_rlzs_per_cosmo)
        
    return np.array(theta_test)
    
    
    

def get_param_names(tag_params=None, tag_biasparams=None, dir_params='../data/params'):
    """
    Gets parameter names given tag_params and tag_biasparams.

    Args:
        tag_params (str): Tag for cosmological parameters.
        tag_biasparams (str): Tag for bias parameters.
        dir_params (str): Directory where parameter files are located.

    Returns:
        list: A list of parameter names.
    """

    param_names = []

    if tag_params is not None:
        if 'fisher' in tag_params:
            fn_params = f'{dir_params}/params{tag_params}.txt'
        else:
            fn_params = f'{dir_params}/params_lh{tag_params}.txt'
        if os.path.exists(fn_params):
            params_df = pd.read_csv(fn_params, index_col=0)
            param_names.extend(params_df.columns.tolist())

    if tag_biasparams is not None:
        fn_biasparams = f'{dir_params}/params_lh{tag_biasparams}.txt'
        if os.path.exists(fn_biasparams):
            biasparams_df = pd.read_csv(fn_biasparams, index_col=0)
            param_names.extend(biasparams_df.columns.tolist())

    return param_names


def load_data_emu(statistic, tag_params, tag_biasparams, tag_errG='', tag_datagen='', tag_noiseless='',
                    n_rlzs_per_cosmo=1, tag_mask=''):
    
    assert statistic=='pk', "Only implemented for pk for emu"
    tag_mocks = tag_params + tag_biasparams

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
    print(fn_emuPk)
    print(Pk.shape)

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
    
    # doing this to align with load_data_muchisimocks; reconsider how to do in emu case
    #idxs_params = np.arange(Pk.shape[0])
    idxs_params = np.array([(idx_LH, idx_LH) for idx_LH in range(Pk.shape[0])])
    # mask = np.all(Pk>0, axis=0)
    # print("mask")
    # print(mask)
    # Pk = Pk[:,mask]
    # gaussian_error_pk = gaussian_error_pk[:,mask]
    # k = k[mask]
    # print(len(k))

    return k, Pk, gaussian_error_pk, idxs_params


# TODO make mask saving more robust
# used to both remove nonpositive data, and to select certain k_bins
def get_Pk_mask(tag_data, tag_mask='', k=None, Pk=None):
    dir_masks = '../data/masks'
    fn_mask = f'{dir_masks}/mask{tag_data}{tag_mask}.txt'
    print(f"fn_mask: {fn_mask}")
    if os.path.exists(fn_mask):
        print(f"Loading from {fn_mask} (already exists)")
        return np.loadtxt(fn_mask, dtype=bool)
    else:
        if Pk is not None:
            mask = np.all(Pk>0, axis=0)
        else:
            assert k is not None, "must pass either Pk or k, if mask doesn't yet exist!"
            print("No Pk provided and no mask exists, using all bins")
            mask = np.ones(len(k), dtype=int)
        print(f"Saving mask to {fn_mask}")
        np.savetxt(fn_mask, mask.astype(int), fmt='%i')
    print(f"Mask masks out {np.sum(~mask)} Pk bins")
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
    if data_mode == 'emu':
        for n_rlz in range(n_rlzs_per_cosmo):
            idxs_train.extend(idxs_train_orig + n_rlz*n_emuPk)
    elif 'muchisimocks' in data_mode:
        idxs_train.extend(idxs_train_orig)
    idxs_train = np.array(idxs_train)
    print(idxs_train.shape)
            
    theta_train, theta_val, theta_test = utils.split_train_val_test(theta, idxs_train, idxs_val, idxs_test)
    y_train, y_val, y_test = utils.split_train_val_test(y, idxs_train, idxs_val, idxs_test)
    y_err_train, y_err_val, y_err_test = utils.split_train_val_test(y_err, idxs_train, idxs_val, idxs_test)

    