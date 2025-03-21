import os
import numpy as np
import itertools
import time
import argparse
from pathlib import Path

import bacco

import data_loader
import utils



def main():
    run_loop()
    #parse_args()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag_params', type=str, default='_quijote_p0_n1000',
                        help='Tag for parameter set')
    parser.add_argument('--idx_LH', type=int, default=0,
                        help='Index of the LH realization')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files')
    parser.add_argument('--n_threads', type=int, default=24,
                        help='Number of threads for power spectrum calculation')
    parser.add_argument('--k_min', type=float, default=0.01,
                        help='Minimum k for power spectrum')
    parser.add_argument('--k_max', type=float, default=0.4,
                        help='Maximum k for power spectrum')
    parser.add_argument('--n_bins', type=int, default=30,
                        help='Number of bins for power spectrum')
    args = parser.parse_args()

    tag_params = args.tag_params
    idx_LH = args.idx_LH
    overwrite = args.overwrite
    n_threads = args.n_threads
    k_min = args.k_min
    k_max = args.k_max
    n_bins = args.n_bins
    
    run(tag_params, idx_LH, overwrite, n_threads, k_min, k_max, n_bins)
    

def run_loop():
    ## main training set
    n_mocks = 10000
    tag_params = f'_p5_n{n_mocks}'
    ## fixed cosmo test set
    #n_mocks = 1000
    #tag_params = f'_quijote_p0_n{n_mocks}'
    ## variable cosmo test set
    #n_mocks = 1000
    #tag_params = f'_test_p5_n{n_mocks}'
    overwrite = False
    n_threads = 24
    k_min = 0.01
    k_max = 0.4
    n_bins = 30

    #idxs_LH = [0]
    idxs_LH = np.arange(n_mocks)
    for idx_LH in idxs_LH:
        run(tag_params, idx_LH, overwrite, n_threads, k_min, k_max, n_bins)    
    
    
def run(tag_params, idx_LH, overwrite, n_threads, k_min, k_max, n_bins):
    tag_biasparams = None #because we're just computing the cross-spectra!

    print("Starting muchisimocks Pnn computation")

    dir_mocks = f'/scratch/kstoreyf/muchisimocks/muchisimocks_lib{tag_params}'
    dir_pnns = f'/scratch/kstoreyf/muchisimocks/data/pnns_mlib/pnns{tag_params}'
    Path.mkdir(Path(dir_pnns), parents=True, exist_ok=True)

    params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, random_ints, random_ints_bias = \
        data_loader.load_params(tag_params, tag_biasparams)

    if 'p0' in tag_params:
        subdir_prefix = 'mock'
    else:
        subdir_prefix = 'LH'

    # We only compute for the specified idx_LH
    fn_fields = f'{dir_mocks}/{subdir_prefix}{idx_LH}/bias_fields_eul_deconvolved_{idx_LH}.npy'
    fn_pnn = f'{dir_pnns}/pnn_{idx_LH}.npy'

    if os.path.exists(fn_pnn) and not overwrite:
        print(f"Pnn for idx_LH={idx_LH} exists and overwrite={overwrite}, exiting")
        return

    start = time.time()
    try:
        bias_terms_eul = np.load(fn_fields)
    except FileNotFoundError:
        print(f"File {fn_fields} not found, exiting")
        return

    n_grid_orig = 512 # we just know this from how we created the fields
    print(f"n_grid_orig = {n_grid_orig}", flush=True)

    param_dict = param_dict_fixed.copy()
    if params_df is not None:
        param_dict.update(params_df.loc[idx_LH].to_dict())
    print(param_dict, flush=True)
    cosmo = utils.get_cosmo(param_dict)

    if '2Gpc' in fn_fields:  # Extract box size from file name
        box_size = 2000.0
    else:
        box_size = 1000.0

    # TODO  check normalizaiton; n_grid_orig used for normalizing tracer
    # field once computed given bias coeffs; so maybe don't need here?
    power_all_terms = compute_pnn_from_bias_fields(bias_terms_eul, cosmo, box_size, n_grid_orig,
                                            k_min=k_min, k_max=k_max, n_bins=n_bins,
                                            n_threads=n_threads)

    np.save(fn_pnn, power_all_terms)
    end = time.time()
    print(f"Computed Pnn for idx_LH={idx_LH} ({fn_pnn}) in time {end-start} s", flush=True)


def compute_pnn_from_bias_fields(bias_terms_eul, cosmo, box_size, n_grid_orig,
                        k_min=0.01, k_max=0.68, n_bins=30,
                        log_binning=True,
                        normalise_grid=False, deconvolve_grid=True,
                        correct_grid=True,
                        n_threads=8):

    print("Computing the 15 PNN cross power spectra")

    n_grid = bias_terms_eul.shape[-1]
    norm = n_grid_orig**3
    
    log_binning = True
    deposit_method = 'cic'
    interlacing = False
    #correct_grid = True
    correct_grid = False #TRYING
    deconvolve_grid = False
    normalise_grid = False

    args_power_grid = {
        # "grid1": None,
        # "grid2": None,
        "normalise_grid1": normalise_grid, #default: False
        "normalise_grid2": normalise_grid, #default: False
        "deconvolve_grid1": deconvolve_grid, #default: False
        "deconvolve_grid2": deconvolve_grid, #default: False
        "ngrid": n_grid,
        "box": box_size,
        "mass1": None,
        "mass2": None,
        "interlacing": interlacing, #default: True
        "deposit_method": deposit_method, #default: "tsc",
        "log_binning": log_binning,
        "kmin": k_min,
        "kmax": k_max,
        "nbins": n_bins,
        "correct_grid": correct_grid,
        #"zspace": False,
        "cosmology": cosmo,
        "pmulti_interp": "polyfit",
        "nthreads": n_threads,
        "compute_correlation": False, #default: True
        "compute_power2d": False, #default: True
        "folds": 1,
        "totalmass1": None,
        "totalmass2": None,
        "jack_error": False,
        "n_jack": None
    }
    
    pknbody_dict = {
        'ngrid': n_grid,
        'min_k': k_min,
        'log_binning': log_binning,
        'log_binning_kmax': k_max,
        'log_binning_nbins': n_bins,
        'interlacing': interlacing,
        'depmethod': deposit_method,
        'correct_grid': correct_grid,
        'folds': 1 #default
    }
    bacco.configuration.update({'number_of_threads': n_threads})
    bacco.configuration.update({'pknbody': pknbody_dict})
    bacco.configuration.update({'pk' : {'maxk' : k_max}})
    bacco.configuration.update({'scaling' : {'disp_ngrid' : n_grid}})

    prod = np.array(list(itertools.combinations_with_replacement(np.arange(len(bias_terms_eul)), r=2)))

    # In check_pnn.py, they use a linear theory power spectrum for correction
    n_grid = 512 # Assuming a default value, could be made a parameter
    lt_k = np.logspace(np.log10(np.pi / box_size), np.log10(2 * np.pi / box_size * n_grid), num=100)
    pk_lpt = bacco.utils.compute_pt_15_basis_terms(cosmo, expfactor=cosmo.expfactor, wavemode=lt_k)

    power_all_terms = []
    for ii in range(0, len(prod)):
        print(f"Computing cross-spectrum {ii+1} of {len(prod)}", flush=True)
        if ii in [1, 5, 9, 12]:
            pk_lt = {'k': lt_k, 'pk': pk_lpt[0][ii], 'pk_nlin': pk_lpt[0][ii], 'pk_lt_log': True}
        elif ii in [2, 3, 4, 7, 8, 11, 13]:
            pk_lt = {'k': lt_k, 'pk': pk_lpt[0][ii], 'pk_nlin': pk_lpt[0][ii], 'pk_lt_log': False}
        else:
            pk_lt = None
        args_power_grid_ii = args_power_grid.copy()
        #args_power_grid_ii['correct_grid'] = False if ii == 11 else True
        
        # passing pk_lt doesn't seem to be doing anything
        # using correct_grid=False always now bc that's what was doing for pk; that way it matches

        power_term = bacco.statistics.compute_crossspectrum_twogrids(grid1=bias_terms_eul[prod[ii, 0]]/norm,
                                                        grid2=bias_terms_eul[prod[ii, 1]]/norm,
                                                        #pk_lt = pk_lt,
                                                        **args_power_grid_ii)
        power_all_terms.append(power_term)

    return power_all_terms


if __name__ == '__main__':
    main()