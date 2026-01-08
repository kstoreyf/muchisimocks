import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import tensorflow as tf

import utils



def load_data(data_mode, statistics, tag_params, tag_biasparams,
              tag_noise=None, tag_Anoise=None,
              tag_data=None,
              kwargs={}):
    
    k, y, y_err = [], [], []
    for statistic in statistics:
        # idxs_params should be the same for all so will just grab last one
        if data_mode == 'muchisimocks':
            k_i, y_i, y_err_i, idxs_params = load_data_muchisimocks(statistic,
                                            tag_params, tag_biasparams, 
                                            tag_noise=tag_noise, tag_Anoise=tag_Anoise, **kwargs)
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
            print(k_i.shape, y_i.shape, y_err_i.shape)

        k.append(k_i)
        y.append(y_i)
        y_err.append(y_err_i)    

    params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, Anoise_df, Anoise_dict_fixed, random_ints, random_ints_bias = load_params(tag_params, tag_biasparams, tag_Anoise=tag_Anoise)
    return k, y, y_err, idxs_params, params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, Anoise_df, Anoise_dict_fixed, random_ints, random_ints_bias


def load_data_ood(data_mode, statistics, tag_mock,
                  tag_data=None,
              kwargs={}):
    """
    Load out-of-distribution (OOD) data for a specific mode and set of statistics.
    """
    k, y, y_err = [], [], []
    for statistic in statistics:
        # idxs_params should be the same for all so will just grab last one
        if data_mode == 'shame':
            k_i, y_i, y_err_i = load_data_shame(statistic, tag_mock, **kwargs)
        else:
            raise ValueError(f"Data mode {data_mode} not recognized!")
        print(f"Loaded {statistic} data with shape {y_i.shape}")
        if tag_data is None:
            print("No tag_data provided, so not masking data")
        else:
            # tag_data is from training data!
            k_i, y_i, y_err_i = mask_data(statistic, tag_data, k_i, y_i, y_err_i)

        k.append(k_i)
        y.append(y_i)
        y_err.append(y_err_i)    

    return k, y, y_err


def load_data_shame(statistic, tag_mock):
    dir_statistics = f'../data/data_shame/data{tag_mock}/{statistic}s'
    fn_statistics = f'{dir_statistics}/{statistic}.npy'
    if not os.path.exists(fn_statistics):
        raise ValueError(f"File {fn_statistics} does not exist!")
    if statistic == 'pk':
        k, stat, error, pk_obj = load_pk(fn_statistics)
    elif statistic == 'bispec':
        k, stat, error, bispec_obj = load_bispec(fn_statistics)
    else:
        raise ValueError(f"Statistic {statistic} not recognized / computed for this dataset (shame, {tag_mock})!")
    print(f"Loaded {statistic} with shape {stat.shape}")
    return k, stat, error


def get_idxs_params(tag_params, tag_biasparams, tag_Anoise=None, n_bias_per_cosmo=1, 
                    modecosmo='lh'):
    """
    For each cosmological parameter index, get the corresponding bias and noise parameter indices.
    Returns a list of (idx_mock, idx_bias, idx_noise) tuples.
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
                # In most cases, noise index follows bias index pattern
                if tag_Anoise is not None and 'p0' not in tag_Anoise:
                    idx_noise = idx_bias
                else:
                    # For fixed noise or no noise, use mock index
                    idx_noise = idx_mock
                idxs_params.append((idx_mock, idx_bias, idx_noise))
        elif modecosmo == 'fisher':
            if params_df.iloc[idx_mock]['param_shifted'] == 'fiducial':
                for idx_bias in biasparams_df.index:
                    if tag_Anoise is not None and 'p0' not in tag_Anoise:
                        idx_noise = idx_bias
                    else:
                        idx_noise = idx_mock
                    idxs_params.append((idx_mock, idx_bias, idx_noise))
            else:
                fiducial_bias_idxs = np.where(biasparams_df['param_shifted'] == 'fiducial')[0]
                for idx_bias in fiducial_bias_idxs:
                    if tag_Anoise is not None and 'p0' not in tag_Anoise:
                        idx_noise = idx_bias
                    else:
                        idx_noise = idx_mock
                    idxs_params.append((idx_mock, idx_bias, idx_noise))
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


def mask_data(statistic, tag_data, k, y, y_err, tag_mask=''):
    if statistic == 'pk':
        mask = get_Pk_mask(tag_data, tag_mask=tag_mask, k=k, Pk=y)
    elif statistic == 'bispec':
        print(k.shape, y.shape)
        mask = get_bispec_mask(tag_data, tag_mask=tag_mask, k=k, bispec=y)
    else:
        print("No mask for this statistic, using all data")
        mask = [True]*y.shape[-1]
        mask = np.array(mask)
    print(f"Masked {np.sum(~np.array(mask).astype(bool))} out of {len(mask)} bins")
    mask = mask.astype(bool)
    if k.ndim == 1:
        k_masked = k[mask]
    elif k.ndim == 2:
        k_masked = k[:,mask]
    else:
        raise ValueError(f"Unexpected k shape: {k.shape}")
    
    if y.ndim == 1:
        return k_masked, y[mask], y_err[mask]
    elif y.ndim == 2:
        return k_masked, y[:,mask], y_err[:,mask]
    else:
        raise ValueError(f"Unexpected y shape: {y.shape}")


# TODO make mask saving more robust
# used to both remove nonpositive data, and to select certain k_bins
def get_Pk_mask(tag_data, tag_mask='', k=None, Pk=None):
    dir_masks = '../data/masks'
    # if tag_mask is None:
    #     tag_mask = ''
    # need tag_data bc we're checking for any negatives and that depends on the data!
    fn_mask = f'{dir_masks}/mask_pk{tag_data}{tag_mask}.txt'
    print(f"fn_mask: {fn_mask}")
    if os.path.exists(fn_mask):
        print(f"Loading from {fn_mask} (already exists)")
        return np.loadtxt(fn_mask, dtype=bool)
    else:
        assert k is not None, "must pass either Pk or k, if mask doesn't yet exist!"
        
        # Start with base mask: keep positive values if Pk provided, otherwise keep all
        if Pk is not None:
            mask = np.all(Pk > 0, axis=0)
        else:
            print("No Pk provided and no mask exists, using all bins")
            mask = np.ones(len(k), dtype=bool)
        # Apply kmax cutoff if specified
        if 'kmaxpk' in tag_mask:
            match = re.search(r'kmaxpk([\d.]+)', tag_mask)
            if match:
                kmax = float(match.group(1))
                mask = mask & (k < kmax)
            else:
                raise ValueError(f"Could not extract kmax value from tag_mask: {tag_mask}")
        
        print(f"Saving mask to {fn_mask}")
        np.savetxt(fn_mask, mask.astype(int), fmt='%i')
    print(f"Mask masks out {np.sum(~mask)} Pk bins")
    return mask


def get_bispec_mask(tag_data, tag_mask='', k=None, bispec=None):
    dir_masks = '../data/masks'
    #fn_mask = f'{dir_masks}/mask{tag_data}{tag_mask}.txt'
    fn_mask = f'{dir_masks}/mask_bispec{tag_data}{tag_mask}.txt'
    if tag_data is None:
        return np.ones(k.shape[1], dtype=int)
    print(f"fn_mask: {fn_mask}")
    if os.path.exists(fn_mask):
        print(f"Loading from {fn_mask} (already exists)")
        return np.loadtxt(fn_mask, dtype=bool)
    else:
        if 'kmaxbispec' in tag_data:
            match = re.search(r'kmaxbispec([\d.]+)', tag_data)
            if match:
                kmax = float(match.group(1))
            else:
                raise ValueError(f"Could not extract kmax value from tag_data: {tag_data}")
            # print(kmax)
            # print(np.array(k).T)
            mask = np.all(np.array(k).T<kmax, axis=1)
        else:
            assert k is not None, "must pass k, if mask doesn't yet exist!"
            print(f"No mask exists for {tag_data} and can't interpret it, using all bins")
            mask = np.ones(k.shape[1], dtype=int)
        print(f"Saving mask to {fn_mask}")
        np.savetxt(fn_mask, mask.astype(int), fmt='%i')
    print(f"Mask masks out {np.sum(~mask.astype(bool))} bispec bins")
    return mask


def load_params(tag_params=None, tag_biasparams=None,
                tag_Anoise=None,
                dir_params='../data/params'):
    
    if tag_params is None:
        params_df = None
        param_dict_fixed = {}
        random_ints = None
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
        
    if tag_Anoise is None:
        Anoise_df = None
        Anoise_dict_fixed = {}
    else:
        Anoise_df, Anoise_dict_fixed = load_Anoise_params(tag_Anoise, dir_params=dir_params)
            
    # TODO figure out what to do about randoms in Anoise case
    # NOTE this if/else is so current code doesn't break if don't give tag_Anoise, 
    # but probs will want to update everywhere
    return params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, Anoise_df, Anoise_dict_fixed, random_ints, random_ints_bias
    
    
def load_params_ood(data_mode, tag_mock, dir_params='../data/params'):
    if data_mode == 'shame':
        # needed this line bc was getting error message
        import bacco
        cosmo_mock = utils.get_cosmo(utils.cosmo_dict_shame)

        param_dict = {}
        for k, v in cosmo_mock.pars.items():
            kname = k
            if k=='sigma8':
                kname = 'sigma8_cold'
            param_dict[kname] = v

        # dividing b2 by 2 bc marcos said there is a mismatch in the definition of b2 bw prob bias and hybrid bias
        if tag_mock == '_nbar0.00011':
            param_dict.update({'b1': 0.52922445, 'b2': 0.13816352/2, 'bs2': -0.21806094, 'bl': -1.0702721})
        elif tag_mock == '_nbar0.00022':
            param_dict.update({'b1': 0.47410742, 'b2': 0.06350746/2, 'bs2': -0.16940883, 'bl': -0.82443643})
        elif tag_mock == '_nbar0.00054':
            param_dict.update({'b1': 0.40209658, 'b2': -0.00958755/2, 'bs2': -0.09669132, 'bl': -0.79150708})
        else:
            raise ValueError(f"tag_mock {tag_mock} not recognized for shame OOD data!")
        #
        # param_dict.update({'b1': 0.47409821, 'b2': 0.06306578, 'bs2': -0.17022439, 'bl': -0.83432633}) #(via marcos)

        return param_dict
    else:
        raise ValueError(f"Data mode {data_mode} not recognized!")
   
    
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


def load_Anoise_params(tag_Anoise, dir_params='../data/params'):
    # NOTE not handling fisher case rn, don't know if will need
    fn_Anoise = f'{dir_params}/params_lh{tag_Anoise}.txt'
    fn_Anoise_fixed = f'{dir_params}/params_fixed{tag_Anoise}.txt'
    Anoise_df = (
        pd.read_csv(fn_Anoise, index_col=0)
        if os.path.exists(fn_Anoise)
        else None
    )
    Anoise_dict_fixed = (
        pd.read_csv(fn_Anoise_fixed).iloc[0].to_dict() 
        if os.path.exists(fn_Anoise_fixed)
        else {}
    )
    if tag_Anoise is not None:
        # "Anoise_dict_fixed" is the pythonic way to check for a non-empty dict
        assert Anoise_df is not None or Anoise_dict_fixed, f"tag_Anoise {tag_Anoise} should have a corresponding Anoise_df or Anoise_dict_fixed!"
    return Anoise_df, Anoise_dict_fixed
    
    
def param_dfs_to_theta(idxs_params, params_df, biasparams_df, Anoise_df=None, n_rlzs_per_cosmo=1):
    """
    Convert parameter DataFrames to a theta array that can be used for inference.
    Handles cases where params_df, biasparams_df, and Anoise_df have different lengths.
    
    Args:
        idxs_params: List of tuples containing parameter indices
        params_df (pd.DataFrame): DataFrame containing cosmological parameters
        biasparams_df (pd.DataFrame): DataFrame containing bias parameters
        Anoise_df (pd.DataFrame): DataFrame containing noise parameters
        n_rlzs_per_cosmo (int): Number of realizations per cosmology
        
    Returns:
        tuple: (theta, param_names)
            - theta: 2D numpy array of parameter values
            - param_names: List of parameter names corresponding to columns in theta
    """
    assert params_df is not None or biasparams_df is not None or Anoise_df is not None, "At least one of params_df, biasparams_df, or Anoise_df must be specified"
    param_names = []

    # Base cases: only one DataFrame provided
    if params_df is None and biasparams_df is None:
        param_names.extend(Anoise_df.columns.tolist())
        # Extract idx_noise from three-tuple (idx_mock, idx_bias, idx_noise)
        theta_noise_orig = np.array([Anoise_df.loc[idx_noise].values for _, _, idx_noise in idxs_params])
        theta_noise = utils.repeat_arr_rlzs(theta_noise_orig, n_rlzs=n_rlzs_per_cosmo)
        return theta_noise, param_names
    elif params_df is None and Anoise_df is None:
        param_names.extend(biasparams_df.columns.tolist())
        # Extract idx_bias from three-tuple (idx_mock, idx_bias, idx_noise)
        theta_bias_orig = np.array([biasparams_df.loc[idx_bias].values for _, idx_bias, _ in idxs_params])
        theta_bias = utils.repeat_arr_rlzs(theta_bias_orig, n_rlzs=n_rlzs_per_cosmo)
        return theta_bias, param_names
    elif biasparams_df is None and Anoise_df is None:
        param_names.extend(params_df.columns.tolist())
        # Extract idx_mock from three-tuple (idx_mock, idx_bias, idx_noise)
        theta_cosmo_orig = np.array([params_df.loc[idx_mock].values for idx_mock, _, _ in idxs_params])
        theta_cosmo = utils.repeat_arr_rlzs(theta_cosmo_orig, n_rlzs=n_rlzs_per_cosmo)
        return theta_cosmo, param_names
    
    # Add parameter names from all DataFrames
    if params_df is not None:
        param_names.extend(params_df.columns.tolist())
    if biasparams_df is not None:
        param_names.extend(biasparams_df.columns.tolist())
    if Anoise_df is not None:
        param_names.extend(Anoise_df.columns.tolist())

    theta = []
    for idx_mock, idx_bias, idx_noise in idxs_params:
        params_combined = []
        
        if params_df is not None:
            cosmo_params = params_df.loc[idx_mock].values
            params_combined.append(cosmo_params)
        
        if biasparams_df is not None:
            bias_params = biasparams_df.loc[idx_bias].values
            params_combined.append(bias_params)
            
        if Anoise_df is not None:
            noise_params = Anoise_df.loc[idx_noise].values
            params_combined.append(noise_params)
        
        theta.append(np.concatenate(params_combined))        
    theta = np.array(theta)
    
    # TODO test/check this part! not using bc haven't been using emulated set
    # Apply realization repetition if needed
    if n_rlzs_per_cosmo > 1:
        theta = utils.repeat_arr_rlzs(theta, n_rlzs=n_rlzs_per_cosmo)
    
    return theta, param_names
    
    
    
def load_theta_test(tag_params_test, tag_biasparams_test, tag_Anoise_test=None,
                    cosmo_param_names_vary=None, bias_param_names_vary=None, noise_param_names_vary=None,
                    n_rlzs_per_cosmo=1):
    # in theory could make this load whichever of the params want, variable or fixed,
    # but for now we want either completely fixed or completely varied, so this is fine
    # TODO NOW i think we might need to handle this case
    
    # get parameter files
    params_df_test, param_dict_fixed_test, biasparams_df_test, biasparams_dict_fixed_test, Anoise_df_test, Anoise_dict_fixed_test, _, _ = load_params(tag_params_test, tag_biasparams_test, tag_Anoise=tag_Anoise_test)

    # load test data
    if 'p0' in tag_params_test and 'p0' in tag_biasparams_test:
        msg = "If all fixed parameters in test set, need to specify which parameters to provide"
        assert cosmo_param_names_vary is not None and bias_param_names_vary is not None, msg
        # if both tests sets are entirely fixed, our theta is just the fixed data, repeated for each observation
        theta_test = [param_dict_fixed_test[pname] for pname in cosmo_param_names_vary]
        theta_test.extend([biasparams_dict_fixed_test[pname] for pname in bias_param_names_vary])
        # Add noise parameters if they exist and are specified
        print("tag_Anoise_test", tag_Anoise_test)
        print(noise_param_names_vary)
        print(Anoise_dict_fixed_test)
        if tag_Anoise_test is not None and 'p0' in tag_Anoise_test and noise_param_names_vary is not None:
            print("here")
            theta_test.extend([Anoise_dict_fixed_test[pname] for pname in noise_param_names_vary])
    else:
        # for when have a LH of varied bias parameters
        idxs_params = get_idxs_params(tag_params_test, tag_biasparams_test, tag_Anoise=tag_Anoise_test)
        theta_test, param_names = param_dfs_to_theta(idxs_params, params_df_test, biasparams_df_test, Anoise_df_test, n_rlzs_per_cosmo=n_rlzs_per_cosmo)
        
    print(len(theta_test))
    return np.array(theta_test)
    

def load_theta_ood(data_mode, tag_mock,
                   cosmo_param_names_vary=None, bias_param_names_vary=None, noise_param_names_vary=None,):
    param_dict = load_params_ood(data_mode, tag_mock)
    theta_test = [param_dict[pname] if pname in param_dict else np.nan for pname in cosmo_param_names_vary]
    theta_test.extend([param_dict[pname] if pname in param_dict else np.nan for pname in bias_param_names_vary])
    # if 'An1' in tag_mock:
    #     A_noise = 1.0
    #     theta_test.extend([A_noise])
    # elif 'nbar' in tag_mock:
    #     nbar = float(tag_mock.split('_nbar')[-1])
    #     nbar_fiducial = 2.2e-4 # TODO get from noise field generation script
    #     A_noise = 1.0 / np.sqrt(nbar/nbar_fiducial)
    #     print(nbar, A_noise)
    #     theta_test.extend([A_noise])
    # else:
    # we dont know true noise params
    theta_test.extend([param_dict[pname] if pname in param_dict else np.nan for pname in noise_param_names_vary])
    return np.array(theta_test)

    

def get_param_names(tag_params=None, tag_biasparams=None, tag_Anoise=None, dir_params='../data/params'):
    """
    Gets parameter names given tag_params and tag_biasparams.

    Args:
        tag_params (str): Tag for cosmological parameters.
        tag_biasparams (str): Tag for bias parameters.
        tag_Anoise (str): Tag for noise parameters.
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

    if tag_Anoise is not None:
        fn_Anoise = f'{dir_params}/params_lh{tag_Anoise}.txt'
        if os.path.exists(fn_Anoise):
            Anoise_df = pd.read_csv(fn_Anoise, index_col=0)
            param_names.extend(Anoise_df.columns.tolist())

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

    
    
def get_dir_statistics(statistic, tag_params, tag_biasparams, tag_noise=None, tag_Anoise=None):
    if tag_biasparams is not None and tag_noise is not None and tag_Anoise is not None:
        # think tag_noise and tag_Anoise will always go together bc you need to know how to modify noise field
        # (tho the combos can vary, that's why they're different)
        tag_mocks = tag_params + tag_biasparams + tag_noise + tag_Anoise
    elif tag_biasparams is not None:
        tag_mocks = tag_params + tag_biasparams
    elif tag_noise is not None and tag_params is None:
        tag_mocks = tag_noise
    else:
        tag_mocks = tag_params
    dir_statistics = f'/scratch/kstoreyf/muchisimocks/data/{statistic}s_mlib/{statistic}s{tag_mocks}'
    return dir_statistics    


# def get_fns_statistic(statistic, idx_mock, tag_params, tag_biasparams, tag_noise, tag_Anoise,
#                        params_df, biasparams_df):
    
#     dir_statistics = get_dir_statistics(statistic, tag_params, tag_biasparams, tag_noise, tag_Anoise)
#     Path.mkdir(Path(dir_statistics), parents=True, exist_ok=True)
    
#     fns_statistics = []
#     # check whether will need precompute later, or can just define in this func
    
#     # deal with noise-only case
#     # compute the statistic of only the noise field!
#     if tag_params is None:
#         assert tag_noise is not None, "If tag_params is None, tag_noise must be provided for noise-only computation"
#         fn_statistic = f'{dir_statistics}/{statistic}_n{idx_mock}.npy'
#         fns_statistics.append(fn_statistic)
#         idxs_bias = None #?
#         idxs_noise = [idx_mock] # use the mock index as the noise field index
            
#     # varied bias param case
#     elif tag_biasparams is not None and 'p0' not in tag_biasparams:                
            
#         # figure out which bias indices to use
#         if 'fisher' in tag_biasparams:
#             idxs_bias = get_bias_indices_for_idx(idx_mock, modecosmo='fisher',
#                                                 params_df=params_df, biasparams_df=biasparams_df)
#         else:
#             factor, longer_df = check_df_lengths(params_df, biasparams_df)
#             assert longer_df == 'bias' or longer_df == 'same', "In non-precomputed mode, biasparams_df should be longer or same length as params_df"
#             idxs_bias = get_bias_indices_for_idx(idx_mock, modecosmo='lh', factor=factor)
        
#         #actually for now we are aligning with cosmo! TODO confirm
#         idxs_noise = [idx_mock]*len(idxs_bias) 
#         #idxs_noise = idxs_bias
#         for idx_bias, idx_noise in zip(idxs_bias, idxs_noise):
#             if tag_Anoise is not None and 'p0' not in tag_Anoise:
#                 # noise field associated w the bias params
#                 # in the case where 1 bias param per cosmo, idx_mock==idx_bias==idx_noise
#                 fn_statistic = f'{dir_statistics}/{statistic}_{idx_mock}_b{idx_bias}_n{idx_noise}.npy' 
#             else:
#                 fn_statistic = f'{dir_statistics}/{statistic}_{idx_mock}_b{idx_bias}.npy'
#             fns_statistics.append(fn_statistic)
    
#     # fixed bias param case, OR pnn (only dep on cosmo)
#     else:
#         #idxs_bias = None # bias not needed, either bc pnn, or noise-only
#         # i believe this is just to run the loop, it will only be cases for fixed biasparams,
#         # or none at all (e.g. pnn)
#         idxs_bias = [0] # because for noise we may also need (?)
        
#         if tag_biasparams is not None and 'p0' in tag_biasparams and tag_noise is not None:
#             # dep on cosmo and noise, e.g. fixed bias params but with noise
#             idx_noise = idx_mock
#             idxs_noise = [idx_noise]
#             fn_statistic = f'{dir_statistics}/{statistic}_{idx_mock}_n{idx_noise}.npy'
#         else:    
#             # only dep on cosmo, e.g. pnn
#             idxs_noise = None
#             fn_statistic = f'{dir_statistics}/{statistic}_{idx_mock}.npy'
#         fns_statistics.append(fn_statistic)
#     return fns_statistics, idxs_bias, idxs_noise


## updated version by claude, incl helper functions below here
def load_data_muchisimocks(statistic, tag_params, tag_biasparams, 
                           tag_noise=None, tag_Anoise=None,
                           tag_datagen='',
                           mode_precomputed=None, return_pk_objs=False):
    
    # Determine directories and file structure
    dir_statistics = get_dir_statistics(statistic, tag_params, tag_biasparams, 
                                        tag_noise=tag_noise, tag_Anoise=tag_Anoise)
    print(f"dir_statistics: {dir_statistics}")
    stat_name = statistic
    
    if os.path.exists(dir_statistics):
        mode_precomputed = True
    else:
        if statistic == 'pk' or statistic == 'pklin':
            stat_name = 'pnn'
            # this was outside the if before, check nothing broke??
            dir_statistics = get_dir_statistics(stat_name, tag_params, None)
        mode_precomputed = False
    
    print(f"Loading muchisimocks data from {dir_statistics}")    
    
    # Load parameter dataframes
    params_df, param_dict_fixed = load_cosmo_params(tag_params)
    biasparams_df, biasparams_dict_fixed = load_bias_params(tag_biasparams)
    Anoise_df, Anoise_dict_fixed = load_Anoise_params(tag_Anoise)

    # Find available cosmologies
    idxs_LH = _get_available_cosmologies(dir_statistics, stat_name)
    print(f"Found {len(idxs_LH)} diff cosmo {stat_name}s in {dir_statistics}")
    assert len(idxs_LH) > 0, f"No {stat_name}s found in {dir_statistics}!"

    # Initialize output arrays
    stat_arr, error_arr, idxs_params = [], [], []
    k = None  # Will be set from first successful cosmology
    if return_pk_objs:
        pk_obj_arr = []

    # Main processing loop - cleaner structure
    for idx_LH in idxs_LH:
        #print(f"idx_LH={idx_LH}")
        cosmology_results = _process_single_cosmology(
            idx_LH, statistic, stat_name, mode_precomputed, dir_statistics,
            params_df, biasparams_df, biasparams_dict_fixed,
            Anoise_df, Anoise_dict_fixed,
            tag_biasparams, tag_noise, tag_Anoise, return_pk_objs
        )
        
        if cosmology_results is None:
            continue
            
        # Unpack results and extend arrays
        cosmo_k, cosmo_stats, cosmo_errors, cosmo_idxs, cosmo_objs = cosmology_results
        
        # Set k from first successful cosmology (should be same for all)
        if k is None:
            k = cosmo_k
        stat_arr.extend(cosmo_stats)
        error_arr.extend(cosmo_errors)
        idxs_params.extend(cosmo_idxs)
        
        if return_pk_objs:
            pk_obj_arr.extend(cosmo_objs)
            
        # print("BREAKING AFTER IDXLH1 TO TEST")
        # break

    stat_arr = np.array(stat_arr)
    error_arr = np.array(error_arr)
    idxs_params = np.array(idxs_params)
        
    if return_pk_objs:
        return k, stat_arr, error_arr, idxs_params, pk_obj_arr
    else:
        return k, stat_arr, error_arr, idxs_params


def _get_available_cosmologies(dir_statistics, stat_name):
    """Extract cosmology indices from available files."""
    idxs_LH = [int(re.search(fr'{stat_name}_(\d+)[^/]*\.npy', file_name).group(1))
               for file_name in os.listdir(dir_statistics) 
               if re.search(fr'{stat_name}_(\d+)[^/]*\.npy', file_name)]
    return np.sort(np.array(list(set(idxs_LH))))


def _process_single_cosmology(idx_LH, statistic, stat_name, mode_precomputed, dir_statistics,
                             params_df, biasparams_df, biasparams_dict_fixed,
                             Anoise_df, Anoise_dict_fixed,
                             tag_biasparams, tag_noise, tag_Anoise, return_pk_objs):
    """Process data for a single cosmology, handling all bias parameter variations."""
    
    if mode_precomputed:
        return _process_precomputed_cosmology(
            idx_LH, statistic, dir_statistics, params_df, biasparams_df,
            tag_biasparams, tag_Anoise, return_pk_objs
        )
    else:
        return _process_pnn_cosmology(
            idx_LH, statistic, stat_name, dir_statistics,
            params_df, biasparams_df, biasparams_dict_fixed,
            Anoise_df, Anoise_dict_fixed,
            tag_noise, tag_Anoise, return_pk_objs
        )


def _process_precomputed_cosmology(idx_LH, statistic, dir_statistics, 
                                  params_df, biasparams_df, tag_biasparams, tag_Anoise, return_pk_objs):
    """Handle precomputed pk/bispec files - original logic."""
    
    # Get filenames and indices using existing logic
    _, fns_statistics, idxs_bias, idxs_noise = get_fns_statistic_precomputed(
        statistic, idx_LH, dir_statistics, params_df, biasparams_df, tag_biasparams, tag_Anoise
    )
    
    stats, errors, indices, objs = [], [], [], []
    k = None
    
    for i, fn_stat in enumerate(fns_statistics):
        if not os.path.exists(fn_stat):
            #print(f"WARNING: Missing {fn_stat}, skipping")
            #continue
            raise FileNotFoundError(f"Statistic file {fn_stat} not found!")
            
        idx_bias = idxs_bias[i] if idxs_bias is not None else None
        idx_noise = idxs_noise[i] if idxs_noise is not None else None
        
        # Load precomputed data
        if statistic == 'pk':
            k_loaded, stat, error, pk_obj = load_pk(fn_stat)
            if k is None:
                k = k_loaded
            if return_pk_objs:
                objs.append(pk_obj)
        elif statistic == 'bispec':
            k_loaded, stat, error, bispec_obj = load_bispec(fn_stat, n_grid=128)
            if k is None:
                k = k_loaded
            if return_pk_objs:
                objs.append(bispec_obj)
        
        stats.append(stat)
        errors.append(error)
        indices.append((idx_LH, idx_bias, idx_noise))
    
    return k, stats, errors, indices, objs


def load_pk(fn_stat):
    pk_obj = np.load(fn_stat, allow_pickle=True).item()
    k, stat, error = pk_obj['k'], pk_obj['pk'], pk_obj['pk_gaussian_error']
    return k, stat, error, pk_obj


def load_bispec(fn_stat, n_grid=None):
    # n_grid used for normalization        
    bispec_obj = np.load(fn_stat, allow_pickle=True).item()
    if 'n_grid' in bispec_obj:
        n_grid_loaded = bispec_obj['n_grid']
        if n_grid is None:
            n_grid = n_grid_loaded
        elif n_grid != n_grid_loaded:
            raise ValueError(f"WARNING: n_grid mismatch in bispec file {fn_stat} (expected {n_grid}, got {n_grid_loaded})")
    else:
        assert n_grid is not None, "n_grid must be provided (or in bispec_obj) if not in bispec file"

    norm = n_grid**3
    k = bispec_obj['k123']
    stat = norm**3 * bispec_obj['weight'] * bispec_obj['bispectrum']['b0']
    error = np.zeros_like(stat)
    return k, stat, error, bispec_obj


def _process_pnn_cosmology(idx_LH, statistic, stat_name, dir_statistics,
                          params_df, biasparams_df, biasparams_dict_fixed,
                          Anoise_df, Anoise_dict_fixed,
                          tag_noise, tag_Anoise, return_pk_objs):
    """Handle pnn files with bias parameter variations - the problematic case."""
    
    # Load the single pnn file for this cosmology
    fn_pnn = f"{dir_statistics}/{stat_name}_{idx_LH}.npy"
    if not os.path.exists(fn_pnn):
        #print(f"WARNING: Missing {fn_pnn}, skipping cosmology")
        #return None
        raise FileNotFoundError(f"PNN file {fn_pnn} not found!")
    
    pnn_obj = np.load(fn_pnn, allow_pickle=True)
    k = pnn_obj[0]['k']
    
    # Determine which bias parameters to use
    idxs_bias = _get_bias_indices_for_cosmology(idx_LH, params_df, biasparams_df)
    
    stats, errors, indices, objs = [], [], [], []
    
    # Process each bias parameter combination
    for idx_bias in idxs_bias:
        # Get bias parameters
        if biasparams_df is not None:
            bias_params_dict = biasparams_df.loc[idx_bias].to_dict()
        else:
            bias_params_dict = {}
        bias_params_dict.update(biasparams_dict_fixed)
        bias_params = [bias_params_dict[bpn] for bpn in utils.biasparam_names_ordered]
        
        # Convert pnn to pk
        if statistic == 'pk':
            stat = utils.pnn_to_pk(pnn_obj, bias_params, pk_type='pk')
        elif statistic == 'pklin':
            stat = utils.pnn_to_pk(pnn_obj, bias_params, pk_type='pk_theory_lin')
        else:
            raise ValueError(f"Statistic {statistic} not supported for pnn mode")
        
        # Handle noise if specified
        if tag_noise is not None:
            stat = _add_noise_to_statistic(stat, idx_LH, idx_bias, tag_noise, tag_Anoise,
                                          Anoise_df, Anoise_dict_fixed)
        
        # Determine noise index - always use idx_LH
        idx_noise = idx_LH
            
        stats.append(stat)
        errors.append(np.zeros_like(stat))
        indices.append((idx_LH, idx_bias, idx_noise))
        
        if return_pk_objs:
            objs.append(pnn_obj)  # Could create a modified object here if needed
    
    return k, stats, errors, indices, objs


def _get_bias_indices_for_cosmology(idx_LH, params_df, biasparams_df):
    """Get the appropriate bias parameter indices for a given cosmology."""
    
    if biasparams_df is None:
        return [0]  # Default case - single bias parameter set
    
    # Check if this is a Fisher matrix case
    if params_df is not None and 'param_shifted' in params_df.columns:
        # Fisher case
        if params_df.iloc[idx_LH]['param_shifted'] == 'fiducial':
            return biasparams_df.index.tolist()
        else:
            return np.where(biasparams_df['param_shifted'] == 'fiducial')[0].tolist()
    else:
        # Latin Hypercube case
        factor, longer_df = check_df_lengths(params_df, biasparams_df)
        if longer_df == 'bias':
            return get_bias_indices_for_idx(idx_LH, modecosmo='lh', factor=factor)
        else:
            return [idx_LH]  # 1-to-1 mapping


def _add_noise_to_statistic(stat, idx_LH, idx_bias, tag_noise, tag_Anoise,
                           Anoise_df, Anoise_dict_fixed):
    """Add noise to a statistic if noise parameters are specified."""
    
    # Load noise power spectrum
    dir_noise = get_dir_statistics('pk', None, None, tag_noise, None)
    fn_noise = f'{dir_noise}/pk_n{idx_LH}.npy'
    
    if not os.path.exists(fn_noise):
        raise FileNotFoundError(f"Noise file {fn_noise} not found!")
    
    pk_obj_noise = np.load(fn_noise, allow_pickle=True).item()
    stat_noise = pk_obj_noise['pk']
    
    # Get noise amplitude - always use idx_LH for noise index
    A_noise_dict = {}
    if Anoise_dict_fixed:
        A_noise_dict.update(Anoise_dict_fixed)
        
    if Anoise_df is not None:
        A_noise_dict.update(Anoise_df.loc[idx_LH].to_dict())
    
    A_noise = A_noise_dict['A_noise']
    
    return stat + A_noise * stat_noise


def get_fns_statistic_precomputed(statistic, idx_mock, dir_statistics, 
                                 params_df, biasparams_df, tag_biasparams, tag_Anoise):
    """
    Get filenames for precomputed statistics (pk, bispec).
    This is the cleaned-up version of the original get_fns_statistic logic.
    """
    
    fns_statistics = []
    
    # For precomputed data, files already exist with bias parameters applied
    if biasparams_df is not None and 'p0' not in tag_biasparams:
        # Varied bias parameters case
        if 'fisher' in str(biasparams_df.index[0]):  # Quick fisher check
            idxs_bias = get_bias_indices_for_idx(idx_mock, modecosmo='fisher',
                                               params_df=params_df, biasparams_df=biasparams_df)
        else:
            factor, longer_df = check_df_lengths(params_df, biasparams_df)
            idxs_bias = get_bias_indices_for_idx(idx_mock, modecosmo='lh', factor=factor)
        
        # Generate filenames for each bias parameter
        idxs_noise = [idx_mock] * len(idxs_bias)  # Align with cosmology
        for idx_bias, idx_noise in zip(idxs_bias, idxs_noise):
            if tag_Anoise is not None and 'p0' not in tag_Anoise:
                fn_stat = f'{dir_statistics}/{statistic}_{idx_mock}_b{idx_bias}_n{idx_noise}.npy'
            else:
                fn_stat = f'{dir_statistics}/{statistic}_{idx_mock}_b{idx_bias}.npy'
            fns_statistics.append(fn_stat)
    else:
        # Fixed bias parameters or no bias parameters
        idxs_bias = [0]  # Single bias parameter set
        # if we're using noise at all, we'll want idx_noise set; 
        # even if fixed A_noise, we'll be using the set of noise fields also indexed by idx_noise
        # (if tag_Anoise is not None, tag_noise is also not None)
        if tag_Anoise is not None:
            # noisy & varying noise params
            idx_noise = idx_mock
            idxs_noise = [idx_noise]
            fn_stat = f'{dir_statistics}/{statistic}_{idx_mock}_n{idx_noise}.npy'
        else:
            idxs_noise = None
            fn_stat = f'{dir_statistics}/{statistic}_{idx_mock}.npy'
        fns_statistics.append(fn_stat)
    
    return dir_statistics, fns_statistics, idxs_bias, idxs_noise


def get_fns_statistic_noise_only(statistic, idx_mock, tag_noise):
    """
    Handle the special case of noise-only statistics.
    Separated out for clarity.
    """
    dir_statistics = get_dir_statistics(statistic, None, None, tag_noise, None)
    fn_statistic = f'{dir_statistics}/{statistic}_n{idx_mock}.npy'
    return dir_statistics, [fn_statistic], None, [idx_mock]


# Optional: Clean wrapper that chooses the right approach
def get_fns_statistic(statistic, idx_mock, tag_params, tag_biasparams, 
                     tag_noise, tag_Anoise, params_df, biasparams_df):
    """
    Simplified wrapper that routes to the appropriate function.
    Only use this for precomputed data or noise-only cases.
    """
    
    # Noise-only case
    if tag_params is None and tag_noise is not None:
        return get_fns_statistic_noise_only(statistic, idx_mock, tag_noise)
    
    # Precomputed case
    dir_statistics = get_dir_statistics(statistic, tag_params, tag_biasparams, 
                                       tag_noise, tag_Anoise)
    dir_statistics, fns_statistics, idxs_bias, idxs_noise = get_fns_statistic_precomputed(statistic, idx_mock, dir_statistics,
                                        params_df, biasparams_df, tag_biasparams, tag_Anoise)
    return dir_statistics, fns_statistics, idxs_bias, idxs_noise