"""
Fit noise model parameters to observed power spectrum and bispectrum.

This script fits noise parameters (A_noise and A2_noise) to match observed
statistics by optimizing over a subset of free parameters.
"""

import gc
import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from scipy.optimize import minimize

import utils
import compute_statistics as cs
import data_loader


# Default parameter bounds
DEFAULT_PARAM_RANGES_A = [
    [0.0, 2.0],   # A_noise[0]
    [0.0, 2.0],   # A_noise[1]
    [0.0, 2.0],   # A_noise[2]
    [0.0, 10.0],   # A_noise[3]
    [0.0, 20.0],  # A_noise[4]
]

DEFAULT_PARAM_RANGES_A2 = [
    [0.0, 0.5],   # A2_noise[0]
    [0.0, 0.5],   # A2_noise[1]
    [0.0, 0.5],   # A2_noise[2]
    [0.0, 10.0],   # A2_noise[3]
    [0.0, 10.0]    # A2_noise[4]
]


def create_fit_config(idxs_Anoise_vary, initial_params=None, As_noise_fixed=None, 
                      A2s_noise_fixed=None, bounds=None, method='TNC', 
                      opt_options=None, use_pk=True, use_bk=True, config_name='default'):
    """Create a fit configuration dict."""
    if initial_params is None:
        initial_params = np.zeros(len(idxs_Anoise_vary))
    else:
        initial_params = np.array(initial_params)
    
    if As_noise_fixed is None:
        As_noise_fixed = [0.0] * 5
    if A2s_noise_fixed is None:
        A2s_noise_fixed = [0.0] * 5
    
    if bounds is None:
        bounds = DEFAULT_PARAM_RANGES_A + DEFAULT_PARAM_RANGES_A2
    
    if opt_options is None:
        if method == 'TNC':
            opt_options = {'maxfun': 150, 'ftol': 1e-1, 'eps': 1e-1, 'disp': True}
        elif method == 'Nelder-Mead':
            opt_options = {'maxiter': 100, 'fatol': 10, 'xatol': 1e-2, 'adaptive': True, 'disp': True}
        elif method == 'L-BFGS-B':
            opt_options = {'maxfun': 50, 'ftol': 1e-2, 'eps': 1e-4, 'disp': True}
        else:
            opt_options = {}
    
    return {
        'idxs_Anoise_vary': idxs_Anoise_vary,
        'initial_params': initial_params,
        'As_noise_fixed': As_noise_fixed,
        'A2s_noise_fixed': A2s_noise_fixed,
        'bounds': bounds,
        'method': method,
        'opt_options': opt_options,
        'use_pk': use_pk,
        'use_bk': use_bk,
        'config_name': config_name,
    }


def create_data_config(data_mode_test='shame', tag_mock='_nbar0.00022', idx_mock=0,
                       dir_mocks='/scratch/kstoreyf/muchisimocks/muchisimocks_lib_ood/shame',
                       subdir_prefix='mock', data_mode_train='muchisimocks',
                       tag_params='_p5_n10000', tag_biasparams='_biaszen_p4_n10000',
                       tag_noise='_noise_unit_p5_n10000', tag_Anoise='_Anmult_p5_n10000',
                       tag_datagen='', tag_mask='', n_grid=128, n_grid_orig=512, box_size=1000.0,
                       seed=42, statistics=None):
    """Create a data configuration dict."""
    if statistics is None:
        statistics = ['pk', 'bispec']
    
    return {
        'data_mode_test': data_mode_test,
        'tag_mock': tag_mock,
        'idx_mock': idx_mock,
        'dir_mocks': dir_mocks,
        'subdir_prefix': subdir_prefix,
        'data_mode_train': data_mode_train,
        'tag_params': tag_params,
        'tag_biasparams': tag_biasparams,
        'tag_noise': tag_noise,
        'tag_Anoise': tag_Anoise,
        'tag_datagen': tag_datagen,
        'tag_mask': tag_mask,
        'n_grid': n_grid,
        'n_grid_orig': n_grid_orig,
        'box_size': box_size,
        'seed': seed,
        'statistics': statistics,
    }


def load_fit_data(data_config: dict):
    """Load test data (OOD mock)."""
    statistics = data_config['statistics']
    # Build tag_data for masking: '_' + data_mode + tag_mask + tag_stats + tag_params + tag_biasparams
    tag_stats = f'_{"_".join(statistics)}'
    tag_mask = data_config['tag_mask'] if 'tag_mask' in data_config else ''
    tag_data = f'_{data_config["data_mode_train"]}{tag_mask}{tag_stats}{data_config["tag_params"]}{data_config["tag_biasparams"]}'
    k_mock, y_mock, y_err_mock = data_loader.load_data_ood(
        data_config['data_mode_test'], statistics, data_config['tag_mock'], tag_data=tag_data
    )
    
    # Get masks from fit data
    k_bispec_mock = k_mock[statistics.index('bispec')]
    y_bispec_mock = y_mock[statistics.index('bispec')]
    
    mask_pk = data_loader.get_Pk_mask(tag_data)
    mask_bispec = data_loader.get_bispec_mask(tag_data, k=k_bispec_mock, bispec=y_bispec_mock)
    
    fn_fields = (f'{data_config["dir_mocks"]}/{data_config["subdir_prefix"]}{data_config["idx_mock"]}/'
                 f'bias_fields_eul_deconvolved_{data_config["idx_mock"]}.npy')
    bias_terms_eul = np.load(fn_fields)
    
    param_dict = data_loader.load_params_ood(data_config['data_mode_test'], data_config['tag_mock'])
    cosmo = utils.get_cosmo(param_dict)
    bias_vector = [param_dict[name] for name in utils.biasparam_names_ordered]
    
    rng = np.random.default_rng(seed=data_config['seed'])
    noise_field_unit = rng.standard_normal((data_config['n_grid'], data_config['n_grid'], data_config['n_grid']))
    
    bs = np.concatenate(([1.0], bias_vector))
    tracer_field_noiseless = np.sum(
        [bs[i] * bias_terms_eul[i] for i in range(len(bs))], axis=0
    ) / data_config['n_grid_orig']**3
    
    gc.collect()
    base_bispec = cs.setup_bispsec(data_config['box_size'], data_config['n_grid'], n_threads=1)
    
    return {
        'k_mock': k_mock,
        'y_mock': y_mock,
        'y_err_mock': y_err_mock,
        'mask_pk': mask_pk,
        'mask_bispec': mask_bispec,
        'bias_terms_eul': bias_terms_eul,
        'cosmo': cosmo,
        'noise_field_unit': noise_field_unit,
        'tracer_field_noiseless': tracer_field_noiseless,
        'base_bispec': base_bispec,
    }


def load_training_data_for_errors(data_config: dict):
    """Load training data to compute error estimates."""
    statistics = data_config['statistics']
    tag_stats = f'_{"_".join(statistics)}'
    # Build tag_data: '_' + data_mode + tag_mask + tag_stats + tag_params + tag_biasparams
    tag_mask = data_config['tag_mask'] if 'tag_mask' in data_config else ''
    tag_data = f'_{data_config["data_mode_train"]}{tag_mask}{tag_stats}{data_config["tag_params"]}{data_config["tag_biasparams"]}'
    
    kwargs_data = {'tag_datagen': data_config['tag_datagen']}
    
    k_arr, y_arr, y_err, idxs_params, params_df, param_dict_fixed, \
    biasparams_df, biasparams_dict_fixed, Anoise_df, Anoise_dict_fixed, \
    random_ints, random_ints_bias = data_loader.load_data(
        data_config['data_mode_train'], statistics,
        data_config['tag_params'], data_config['tag_biasparams'],
        tag_noise=data_config['tag_noise'], tag_Anoise=data_config['tag_Anoise'],
        tag_data=tag_data, kwargs=kwargs_data
    )
    
    mask_pk = data_loader.get_Pk_mask(tag_data)
    
    # Unpack statistics
    k_pk = k_arr[statistics.index('pk')]
    y_pk = y_arr[statistics.index('pk')]
    k_bispec = k_arr[statistics.index('bispec')]
    y_bispec = y_arr[statistics.index('bispec')]
    
    # Get bispectrum mask (needs k and bispec data)
    mask_bispec = data_loader.get_bispec_mask(tag_data, k=k_bispec, bispec=y_bispec)
    
    # Compute error estimates from percentiles
    pk_mean = np.mean(y_pk, axis=0)
    pk_p16 = np.percentile(y_pk, 16, axis=0)
    pk_p84 = np.percentile(y_pk, 84, axis=0)
    pk_err = (pk_p84 - pk_p16) / 2
    
    bispec_mean = np.mean(y_bispec, axis=0)
    bispec_p16 = np.percentile(y_bispec, 16, axis=0)
    bispec_p84 = np.percentile(y_bispec, 84, axis=0)
    bispec_err = (bispec_p84 - bispec_p16) / 2
    
    return {
        'mask_pk': mask_pk,
        'mask_bispec': mask_bispec,
        'pk_err': pk_err,
        'bispec_err': bispec_err,
    }


def fit_noise_parameters(pk_obs, bk_obs, pk_err, bk_err, mask_pk, mask_bispec, bias_terms_eul, 
                             tracer_field_noiseless, noise_field_unit, cosmo, box_size, n_grid,
                         n_grid_orig, base_bispec, fit_config: dict):
    """
    Fit noise parameters to observed pk and bk.
    
    Parameters:
    -----------
    pk_obs : array
        Observed power spectrum
    bk_obs : array
        Observed bispectrum
    pk_err : array
        Power spectrum error estimates
    bk_err : array
        Bispectrum error estimates
    mask_pk : array
        Mask for power spectrum k-bins
    mask_bispec : array
        Mask for bispectrum k-bins
    bias_terms_eul : array
        Eulerian bias terms
    tracer_field_noiseless : array
        Noiseless tracer field
    noise_field_unit : array
        Unit noise field
    cosmo : object
        Cosmology object
    box_size : float
        Box size
    n_grid : int
        Grid size for computation
    n_grid_orig : int
        Original grid size
    base_bispec : object
        Bispectrum base object
    fit_config : dict
        Configuration for fitting
        
    Returns:
    --------
    dict : Results dictionary with fitted parameters and model statistics
    """
    gc.collect()
    
    # Set fixed values
    As_noise_fixed = np.array(fit_config['As_noise_fixed'])
    A2s_noise_fixed = np.array(fit_config['A2s_noise_fixed'])
    
    def params_to_full(params_vary):
        """Convert varying parameters to full 10-parameter array."""
        params_full = np.zeros(10)
        params_full[:5] = As_noise_fixed.copy()
        params_full[5:] = A2s_noise_fixed.copy()
        for idx, val in zip(fit_config['idxs_Anoise_vary'], params_vary):
            params_full[idx] = val
        return params_full
    
    def compute_model_statistics(params_vary):
        """Compute model pk and bk for given varying parameters."""
        params_full = params_to_full(params_vary)
        As_noise = params_full[:5]
        A2s_noise = params_full[5:]
        
        # Create tracer field
        tracer_field_noise = np.sum(
            [As_noise[i] * noise_field_unit * bias_terms_eul[i]
             for i in range(len(As_noise))], axis=0
        ) / n_grid_orig**3
        tracer_field_noise2 = np.sum(
            [A2s_noise[i] * noise_field_unit**2 * bias_terms_eul[i]
             for i in range(len(A2s_noise))], axis=0
        ) / n_grid_orig**3
        tracer_field = tracer_field_noiseless + tracer_field_noise + tracer_field_noise2
        
        pk_obj, bspec, bk_corr = None, None, None
        if fit_config['use_pk']:
            pk_obj = cs.compute_pk(tracer_field, cosmo, box_size, n_threads=1, fn_stat=None)
        if fit_config['use_bk']:
            bspec, bk_corr = cs.compute_bispectrum(base_bispec, tracer_field)
        
        return pk_obj, bspec, bk_corr
    
    def objective(params_vary):
        """Objective function to minimize."""
        pk_obj, bspec, bk_corr = compute_model_statistics(params_vary)
        chi2 = 0.0
        
        if fit_config['use_pk'] and pk_obj is not None:
            pk_model = pk_obj['pk']
            diff = (pk_model[mask_pk] - pk_obs)**2 / pk_err**2
            chi2 += np.sum(diff)
        
        if fit_config['use_bk'] and bk_corr is not None:
            k123 = bspec.get_ks()
            weight = k123.prod(axis=0)
            norm = n_grid**3
            bk_model = norm**3 * weight * bk_corr['b0']
            diff = (bk_model[mask_bispec] - bk_obs)**2 / bk_err**2
            chi2 += np.sum(diff)
        
        print(f"params_vary: {params_vary}, chi2: {chi2:.6f}")
        return chi2
    
    # Perform optimization
    print(f"Starting optimization with method: {fit_config['method']}")
    print(f"Varying parameters at indices: {fit_config['idxs_Anoise_vary']}")
    print(f"Initial params: {fit_config['initial_params']}")
    
    bounds_vary = np.array(fit_config['bounds'])[fit_config['idxs_Anoise_vary']]
    result_opt = minimize(
        objective, fit_config['initial_params'], method=fit_config['method'],
        bounds=bounds_vary, options=fit_config['opt_options']
    )
    
    # Compute final model statistics
    pk_obj_final, bspec_final, bk_corr_final = compute_model_statistics(result_opt.x)
    params_full = params_to_full(result_opt.x)
    
    # Extract pickable parts from bispectrum objects in same format as save_bispectrum
    bispec_results_dict = None
    if bspec_final is not None and bk_corr_final is not None:
        try:
            k123 = bspec_final.get_ks()
            weight = k123.prod(axis=0)
            bispec_results_dict = {
                'k123': k123,
                'bispectrum': bk_corr_final,
                'weight': weight,
                'n_grid': n_grid,
            }
        except:
            pass
    
    # Prepare output - unpack needed values from result_opt and bispec objects
    result = {
        'params': params_full,
        'params_vary': result_opt.x,
        'As_noise': params_full[:5],
        'A2s_noise': params_full[5:],
        'idxs_vary': fit_config['idxs_Anoise_vary'],
        # Unpacked from result_opt
        'opt_fun': result_opt.fun,
        'success': result_opt.success,
        'message': result_opt.message,
        'opt_nit': result_opt.nit if hasattr(result_opt, 'nit') else None,
        'opt_nfev': result_opt.nfev if hasattr(result_opt, 'nfev') else None,
        'opt_x': result_opt.x,
        # Keep pk_obj_model as-is
        'pk_obj_model': pk_obj_final,
        # Bispectrum results in same format as save_bispectrum
        'bispec_results_dict': bispec_results_dict,
        'config': fit_config,
    }
    
    return result


def save_results(result: dict, data_config: dict, output_dir: str = '../results/noise_fits'):
    """
    Save fitting results to disk.
    
    Parameters:
    -----------
    result : dict
        Results dictionary from fit_noise_parameters
    data_config : DataConfig
        Data configuration used
    output_dir : str
        Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename from config
    config_name = result['config']['config_name']
    idxs_str = '_'.join(map(str, result['idxs_vary']))
    fn_results = output_path / f'noise_fit_{config_name}_vary{idxs_str}.npy'
    
    # Prepare full data to save (configs are already dicts)
    save_data = {
        'result': result,
        'data_config': data_config,
    }
    
    # Save using numpy with allow_pickle for complex objects
    np.save(fn_results, save_data, allow_pickle=True)
    print(f"Saved results to {fn_results}")
    
    # Also save a human-readable summary
    fn_summary = output_path / f'noise_fit_{config_name}_vary{idxs_str}_summary.txt'
    with open(fn_summary, 'w') as f:
        f.write(f"Noise Model Fit Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Configuration: {config_name}\n")
        f.write(f"Varying parameters (indices): {result['idxs_vary']}\n")
        f.write(f"Fitted A_noise: {result['As_noise']}\n")
        f.write(f"Fitted A2_noise: {result['A2s_noise']}\n")
        f.write(f"\nFull parameter array:\n{result['params']}\n")
        f.write(f"Optimization method: {result['config']['method']}\n")
        f.write(f"Success: {result['success']}\n")
        f.write(f"Message: {result['message']}\n")
        f.write(f"Number of function evaluations: {result['opt_nfev']}\n")
        f.write(f"Number of iterations: {result['opt_nit']}\n")
        f.write(f"\nChi^2: {result['opt_fun']:.6f}\n")
    
    return str(fn_results)


def load_results(fn_results: str):
    """
    Load saved fitting results.
    
    Parameters:
    -----------
    fn_results : str
        Path to saved results file
        
    Returns:
    --------
    dict : Dictionary with 'result' and 'data_config' keys
    """
    data = np.load(fn_results, allow_pickle=True).item()
    
    result = data['result']
    data_config = data['data_config']
    
    # Reconstruct bk_corr_model and bspec_model from bispec_results_dict
    if 'bispec_results_dict' in result and result['bispec_results_dict'] is not None:
        bispec_dict = result['bispec_results_dict']
        result['bk_corr_model'] = bispec_dict.get('bispectrum')
        
        # Reconstruct bspec_model wrapper from k123
        k123 = bispec_dict.get('k123')
        if k123 is not None:
            class BspecWrapper:
                def __init__(self, k123):
                    self._k123 = k123
                def get_ks(self):
                    return self._k123
            result['bspec_model'] = BspecWrapper(k123)
            result['k123'] = k123  # Also store directly for convenience
    
    print(f"Loaded results from {fn_results}")
    print(f"Configuration: {result['config']['config_name']}")
    print(f"Fitted A_noise: {result['As_noise']}")
    print(f"Fitted A2_noise: {result['A2s_noise']}")
    
    return {
        'result': result,
        'data_config': data_config,
    }


def main(data_config: dict, fit_config: dict, save_output: bool = True):
    """
    Main function to run noise parameter fitting.
    
    Parameters:
    -----------
    data_config : dict
        Configuration for data loading
    fit_config : dict
        Configuration for fitting
    save_output : bool
        Whether to save results to disk
        
    Returns:
    --------
    dict : Fitting results
    """
    print("Loading test data...")
    test_data = load_fit_data(data_config)
    
    print("Loading training data for error estimates...")
    error_data = load_training_data_for_errors(data_config)
    
    print("Fitting noise parameters...")
    result = fit_noise_parameters(
            pk_obs=test_data['y_mock'][0],
            bk_obs=test_data['y_mock'][1],
            pk_err=error_data['pk_err'],
            bk_err=error_data['bispec_err'],
            mask_pk=test_data['mask_pk'],
            mask_bispec=test_data['mask_bispec'],
            bias_terms_eul=test_data['bias_terms_eul'],
            tracer_field_noiseless=test_data['tracer_field_noiseless'],
            noise_field_unit=test_data['noise_field_unit'],
            cosmo=test_data['cosmo'],
            box_size=data_config['box_size'],
            n_grid=data_config['n_grid'],
            n_grid_orig=data_config['n_grid_orig'],
            base_bispec=test_data['base_bispec'],
            fit_config=fit_config
        )
        
    print(f"\nFitting completed!")
    print(f"Fitted A_noise: {result['As_noise']}")
    print(f"Fitted A2_noise: {result['A2s_noise']}")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Number of function evaluations: {result['opt_nfev']}")
    print(f"Number of iterations: {result['opt_nit']}")
    print(f"Chi^2: {result['opt_fun']:.6f}")
        
    if save_output:
        fn_saved = save_results(result, data_config)
        print(f"Results saved to: {fn_saved}")
    
    return result


if __name__ == '__main__':
    # Example configurations - easy to modify or add new ones
    
    # # Configuration 1: Vary first 4 A_noise parameters
    # config1 = FitConfig(
    #     idxs_Anoise_vary=[0, 1, 2, 3],
    #     initial_params=np.array([0.5, 0.5, 0.5, 0.5]),
    #     config_name='A_noise_0-3'
    # )
    
    # # Configuration 2: Vary A_noise[0,1] and A2_noise[0,1]
    # config2 = FitConfig(
    #     idxs_Anoise_vary=[0, 1, 5, 6],
    #     initial_params=np.array([0.5, 0.5, 0.1, 0.1]),
    #     config_name='A_0-1_A2_0-1'
    # )
    
    # Configuration 3: Vary all A_noise parameters
    # config = FitConfig(
    #     idxs_Anoise_vary=[0, 1, 2, 3, 4],
    #     initial_params=np.array([0.25, 0.25, 0.25, 0.25, 2.5]),
    #     config_name='A_noise_all'
    # )

    config = create_fit_config(
        # idxs_Anoise_vary=[0, 1, 2, 3, 4],
        # initial_params=np.array([0.25, 0.25, 0.25, 0.25, 2.5]),
        # idxs_Anoise_vary=[0, 1],
        # initial_params=np.array([0.25, 0.25]),
        idxs_Anoise_vary=[0, 1, 5, 6],
        initial_params=np.array([0.25, 0.25, 0.25, 0.25]),
        #initial_params=np.array([1.28836744, 0.05393767, 0.1, 0.1]),
        # idxs_Anoise_vary=[0, 1, 4, 5, 6, 9],
        # initial_params=np.array([0.25, 0.25, 2.5, 0.25, 0.25, 2.5]),
        # idxs_Anoise_vary=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        # initial_params=np.array([0.25, 0.25, 0.25, 0.25, 2.5, 0.25, 0.25, 0.25, 0.25, 2.5]),
        # idxs_Anoise_vary=[0, 3, 4, 5, 9],
        # initial_params=np.array([1.0, 0.25, 2.5, 0.25, 2.5]),
        config_name='A_noise_kmaxbispec0.2_TNC'
    )
    
    # Data configuration
    data_config = create_data_config(tag_mask='_kmaxbispec0.2')
    
    # Run with desired configuration
    result = main(data_config, config, save_output=True)