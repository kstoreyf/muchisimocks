import os
import numpy as np
import itertools
import time
import argparse
from pathlib import Path

import bacco
import PolyBin3D as pb

import data_loader
import utils

"""
This script computes statistics (e.g., PNN or bispectrum) for mock data.

Some example commands:
python compute_statistics.py --statistic bispec --idx_mock_start 0 --idx_mock_end 1 --tag_params _quijote_p0_n1000 --tag_biasparams _b1000_p0_n1

"""


def main():
    #run_loop()
    parse_args()
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--statistic', type=str,
                        help='statistic ("pnn" or "bispec")')
    parser.add_argument('--tag_params', type=str, default='_quijote_p0_n1000',
                        help='Tag for parameter set')
    parser.add_argument('--tag_biasparams', type=str, default=None,
                        help='Tag for bias parameter set')
    parser.add_argument('--idx_mock_start', type=int,
                        help='Index of the LH realization to start')
    parser.add_argument('--idx_mock_end', type=int,
                        help='Index of the LH realization to end')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing files')
    parser.add_argument('--n_threads', type=int, default=1,
                        help='Number of threads for calculation')
    args = parser.parse_args()

    statistic = args.statistic
    tag_params = args.tag_params
    tag_biasparams = args.tag_biasparams
    overwrite = args.overwrite
    n_threads = args.n_threads
    
    assert statistic in ['pnn', 'bispec'], "statistic must be 'pnn' or 'bispec'"
    
    if statistic == 'bispec':
        #magic
        box_size = 1000.0
        n_grid = 128
        base = setup_bispsec(box_size, n_grid, n_threads)
    else:
        base = None
        
    print(args.idx_mock_start, args.idx_mock_end)
    for idx_mock in range(args.idx_mock_start, args.idx_mock_end):
        print(idx_mock)
        run(statistic, tag_params, idx_mock,
            overwrite=overwrite, n_threads=n_threads, 
            tag_biasparams=tag_biasparams,
            base_bispec=base)
        

def run_loop():
    ## main training set
    #n_mocks = 10000
    #tag_params = f'_p5_n{n_mocks}'    
    #tag_biasparams = '_biaszen_p4_n10000'

    ## fixed cosmo test set
    #n_mocks = 1000
    #tag_params = f'_quijote_p0_n{n_mocks}'
    ## variable cosmo test set
    #n_mocks = 1000
    #tag_params = f'_test_p5_n{n_mocks}'

    ## fisher
    tag_params = '_fisher_quijote'
    #tag_biasparams = None # for pnn, don't need biasparams
    tag_biasparams = '_fisher_biaszen'
    n_mocks = 21 #magic, i just know this number for fisher; TODO read it in
    
    overwrite = False
    n_threads = 1

    box_size = 1000.0
    n_grid = 128
    
    statistic = 'pnn'
    #statistic = 'bispec'
    if statistic == 'bispec':
        base = setup_bispsec(box_size, n_grid, n_threads)
    else:
        base = None

    #idxs_LH = [0]
    idxs_LH = np.arange(n_mocks) 
    for idx_mock in idxs_LH:
        run(statistic, tag_params, idx_mock, 
            overwrite=overwrite, n_threads=n_threads, 
            tag_biasparams=tag_biasparams,
            base_bispec=base,
            )    
    

def setup_bispsec(box_size, n_grid, n_threads):
    # Load the PolyBin3D class 
    start = time.time()
    base = pb.PolyBin3D([box_size, box_size, box_size], n_grid, 
                    #boxcenter=[0,0,0], # center of the simulation volume
                    #pixel_window='interlaced-tsc', # pixel window function
                    backend='fftw', # backend for performing FFTs ('fftw' for cpu, 'jax' for gpu)
                    nthreads=n_threads, # number of CPUs for performing FFTs (only applies to 'fftw' backend)
                    sightline='global', # line-of-sight [global = z-axis, local = relative to pair]
                    )
    end = time.time()
    print(f"PolyBin3D setup time: {end-start:.2f} s", flush=True)
    return base
    
    
def run(statistic, tag_params, idx_mock, overwrite=False, n_threads=1, 
        tag_biasparams=None,
        base_bispec=None):

    if statistic == 'pnn':
        precompute = False
    if statistic == 'bispec':
        precompute = True
        assert base_bispec is not None, "base_bispec must be provided for bispectrum computation"

    print(f"Starting muchisimocks {statistic} computation for idx_mock={idx_mock}", flush=True)

    if tag_biasparams is not None:
        tag_mocks = tag_params + tag_biasparams
    else:
        tag_mocks = tag_params
    # actual mock data dir is just cosmology-dependent
    dir_mocks = f'/scratch/kstoreyf/muchisimocks/muchisimocks_lib{tag_params}'
    dir_statistics = f'/scratch/kstoreyf/muchisimocks/data/{statistic}s_mlib/{statistic}s{tag_mocks}_rerunjobarr'
    Path.mkdir(Path(dir_statistics), parents=True, exist_ok=True)

    params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, _, _ = \
        data_loader.load_params(tag_params, tag_biasparams)

    if 'p0' in tag_params or 'fisher' in tag_params:
        subdir_prefix = 'mock'
    else:
        subdir_prefix = 'LH'

    # We only compute for the specified idx_mock
    fn_fields = f'{dir_mocks}/{subdir_prefix}{idx_mock}/bias_fields_eul_deconvolved_{idx_mock}.npy'
    
    # Checking here if we have remaining statistics to compute and if we need to load the fields at all
    # (because loading fields takes some time, so want to check first if we'll need them;
    # we'll also check later for the bispec case if we need to compute that particular one)
    if precompute:                
        # figure out which bias indices to use
        if 'fisher' in tag_biasparams:
            idxs_bias = data_loader.get_bias_indices_for_idx(idx_mock, modecosmo='fisher',
                                                params_df=params_df, biasparams_df=biasparams_df)
        elif 'p0' in tag_biasparams:
            idxs_bias = np.array([0]) #not really, but so loops don't break; dont use besides for looping
        else:
            factor, longer_df = data_loader.check_df_lengths(params_df, biasparams_df)
            assert longer_df == 'bias' or longer_df == 'same', "In non-precomputed mode, biasparams_df should be longer or same length as params_df"
            idxs_bias = data_loader.get_bias_indices_for_idx(idx_mock, modecosmo='lh', factor=factor)
        
        exist_all = True
        for idx_bias in idxs_bias:
            if 'p0' in tag_biasparams:
                fn_statistic = f'{dir_statistics}/{statistic}_{idx_mock}.npy'
            else:
                fn_statistic = f'{dir_statistics}/{statistic}_{idx_mock}_b{idx_bias}.npy'
            if not os.path.exists(fn_statistic) or overwrite:
                exist_all = False
                print(f"At least one statistic for '{dir_statistics}/{statistic}_{idx_mock} (idx_bias={idx_bias}) does not exist and overwrite={overwrite}, so continuing the computation")
                break
        if exist_all:
            print(f"All statistics for '{dir_statistics}/{statistic}_{idx_mock} exists and overwrite={overwrite}, exiting")
            return
    else:
        fn_statistic = f'{dir_statistics}/{statistic}_{idx_mock}.npy'
        if os.path.exists(fn_statistic) and not overwrite:
            print(f"Statistic {fn_statistic} exists and overwrite={overwrite}, exiting")
            return
        else:
            print(f"Computing statistic {fn_statistic} (overwrite={overwrite})")

    start_tot = time.time()
    
    try:
        bias_terms_eul = np.load(fn_fields)
    except FileNotFoundError:
        print(f"File {fn_fields} not found, exiting")
        return

    n_grid_orig = 512 # we just know this from how we created the fields
    print(f"n_grid_orig = {n_grid_orig}", flush=True)

    param_dict = param_dict_fixed.copy()
    if params_df is not None:
        param_dict.update(params_df.loc[idx_mock].to_dict())
    print(param_dict, flush=True)
    cosmo = utils.get_cosmo(param_dict)

    if '2Gpc' in fn_fields:  # Extract box size from file name
        box_size = 2000.0
    else:
        box_size = 1000.0

    # TODO  check normalizaiton; n_grid_orig used for normalizing tracer
    # field once computed given bias coeffs; so maybe don't need here?
    if statistic == 'pnn':
        start = time.time()
        power_all_terms = compute_pnn_from_bias_fields(bias_terms_eul, cosmo, box_size, n_grid_orig,
                                                n_threads=n_threads)
        np.save(fn_statistic, power_all_terms)
        end = time.time()
        print(f"Computed {statistic} for idx_mock={idx_mock} ({fn_statistic}) in time {end-start:.2f} s", flush=True)

    if statistic == 'bispec':
        
        assert tag_biasparams is not None, "tag_biasparams must be provided for bispectrum computation"
        
        # idxs_bias is gotten above, bc used it to check if files exist already
        print(idxs_bias)
        for idx_bias in idxs_bias:
            start = time.time()
            # check if this particular stat already computed
            if 'p0' in tag_biasparams:
                fn_statistic = f'{dir_statistics}/{statistic}_{idx_mock}.npy'
            else:
                fn_statistic = f'{dir_statistics}/{statistic}_{idx_mock}_b{idx_bias}.npy'
            if os.path.exists(fn_statistic) and not overwrite:
                print(f"Statistic {fn_statistic} exists and overwrite={overwrite}, continuing")
                continue
            
            biasparam_dict = biasparams_dict_fixed.copy()
            if biasparams_df is not None:
                biasparam_dict.update(biasparams_df.loc[idx_bias].to_dict())
            bias_vector = [biasparam_dict[name] for name in utils.biasparam_names_ordered]
            print(idx_bias)
            print(bias_vector)
            tracer_field = utils.get_tracer_field(bias_terms_eul, bias_vector, n_grid_norm=n_grid_orig)
            
            bspec, bk_corr = compute_bispectrum(base_bispec, tracer_field)
            k123 = bspec.get_ks()
            weight = k123.prod(axis=0)
            bispec_results_dict = {
                'k123': k123,
                'bispectrum': bk_corr,
                'weight': weight,
            }
            print('k123', k123)
            print('weight', weight)
            print('bk_corr', bk_corr['b0'])

            np.save(fn_statistic, bispec_results_dict)
        
            end = time.time()
            print(f"Computed {statistic} for idx_mock={idx_mock}, idx_bias={idx_bias} ({fn_statistic}) in time {end-start:.2f} s", flush=True)
            if idxs_bias==1:
                print("HIT IDX BIAS=1, BREAKING FOR TESTING")
                break
            
    end_tot = time.time()
    print(f"Total time to compute {statistic}(s) for idx_mock={idx_mock} in time {end_tot-start_tot:.2f} s", flush=True)


def compute_pnn_from_bias_fields(bias_terms_eul, cosmo, box_size, n_grid_orig,
                        n_threads=1):

    print("Computing the 15 PNN cross power spectra")

    k_min = 0.01
    k_max = 0.4
    n_bins = 30
    
    n_grid = bias_terms_eul.shape[-1]
    norm = n_grid_orig**3
    
    log_binning = True
    deposit_method = 'cic'
    interlacing = False
    correct_grid = False 
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




def compute_bispectrum(base, tracer_field):

    k_min = 0.01
    k_max = 0.4
    n_bins = 7
        
    k_edges = np.linspace(k_min, k_max, n_bins+1) # 7 bins
    k_edges_squeeze = k_edges.copy()
    lmax = 1
    
    # Load the bispectrum class
    bspec = pb.BSpec(base, 
                    k_edges, # one-dimensional bin edges
                    applySinv = None, # weighting function [only needed for unwindowed estimators]
                    mask = None, # real-space mask
                    lmax = lmax, # maximum Legendre multipole
                    k_bins_squeeze = k_edges_squeeze, # squeezed bins
                    include_partial_triangles = False, # whether to include bins whose centers do not satisfy triangle conditions
                    )
    
    bk_corr = bspec.Bk_ideal(tracer_field, discreteness_correction=True)

    return bspec, bk_corr


if __name__ == '__main__':
    main()