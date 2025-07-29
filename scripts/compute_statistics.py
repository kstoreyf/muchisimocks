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
    run_loop()
    #parse_args()
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--statistic', type=str,
                        help='statistic ("pnn" or "bispec")')
    parser.add_argument('--tag_params', type=str, default='_quijote_p0_n1000',
                        help='Tag for parameter set')
    parser.add_argument('--tag_biasparams', type=str, default=None,
                        help='Tag for bias parameter set')
    parser.add_argument('--tag_noise', type=str, default=None,
                        help='Tag for noise fields')
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
    tag_noise = args.tag_noise
    overwrite = args.overwrite
    n_threads = args.n_threads
    
    stats_supported = ['pnn', 'pk', 'pklin', 'bispec']
    assert statistic in stats_supported, f"statistic {statistic} not recognized, should be one of {stats_supported}"
    if statistic == 'pnn':
        assert tag_biasparams is None, "you shouldn't be providing tag_biasparams for pnn, it's only computed for the cosmologies! bias param computation is done at dataloading time"
    
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
            tag_noise=tag_noise,
            base_bispec=base)
        

def run_loop():
    ## main training set
    #n_mocks = 10000
    #tag_params = f'_p5_n{n_mocks}'    
    #tag_biasparams = '_biaszen_p4_n10000'

    ## fixed cosmo test set
    n_mocks = 1000
    tag_params = f'_quijote_p0_n{n_mocks}'
    tag_biasparams = '_b1000_p0_n1' # for pnn, don't need biasparams
    #tag_biasparams = None # for pnn, don't need biasparams
    tag_noise = '_noise_quijote_p0_n1000'
    tag_Anoise = '_An1_p0_n1'
    ## variable cosmo test set
    #n_mocks = 1000
    #tag_params = f'_test_p5_n{n_mocks}'

    ## fisher
    #tag_params = '_fisher_quijote'
    #tag_biasparams = None # for pnn, don't need biasparams
    #tag_biasparams = '_fisher_biaszen'
    #n_mocks = 21 #magic, i just know this number for fisher; TODO read it in
    
    overwrite = False
    n_threads = 1

    box_size = 1000.0
    n_grid = 128
    
    #statistic = 'pnn'
    #statistic = 'pklin'
    #statistic = 'bispec'
    statistic = 'pk'
    if statistic == 'bispec':
        base = setup_bispsec(box_size, n_grid, n_threads)
    else:
        base = None

    idxs_LH = [0]
    #idxs_LH = np.arange(n_mocks) 
    for idx_mock in idxs_LH:
        run(statistic, tag_params, idx_mock, 
            overwrite=overwrite, n_threads=n_threads, 
            tag_biasparams=tag_biasparams,
            tag_noise=tag_noise, tag_Anoise=tag_Anoise,
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
        tag_biasparams=None, tag_noise=None, tag_Anoise=None, 
        base_bispec=None):

    print(f"Starting muchisimocks {statistic} computation for idx_mock={idx_mock}", flush=True)

    if tag_biasparams is not None and tag_noise is not None and tag_Anoise is not None:
        # will tag_noise and tag_Anoise always go together??
        tag_mocks = tag_params + tag_biasparams + tag_noise + tag_Anoise
    elif tag_biasparams is not None:
        tag_mocks = tag_params + tag_biasparams
    else:
        tag_mocks = tag_params

    params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, Anoise_df, Anoise_dict_fixed, _, _ = \
        data_loader.load_params(tag_params, tag_biasparams, tag_Anoise)

    if 'p0' in tag_params or 'fisher' in tag_params:
        subdir_prefix = 'mock'
    else:
        subdir_prefix = 'LH'
    dir_mocks = f'/scratch/kstoreyf/muchisimocks/muchisimocks_lib{tag_params}'
    fn_fields = f'{dir_mocks}/{subdir_prefix}{idx_mock}/bias_fields_eul_deconvolved_{idx_mock}.npy'
    
    # these are the filenames we'll produce; for each cosmo, might be multiple bias params
    # (if just one, will be an array of length 1)
    fns_statistics, idxs_bias = get_fns_to_compute(statistic, idx_mock, tag_mocks, tag_biasparams, tag_Anoise,
                                                   params_df, biasparams_df)
    
    # check which statistics we need to compute
    for fn_statistic in fns_statistics:
        if os.path.exists(fn_statistic) and not overwrite:
            print(f"Statistic {fn_statistic} exists and overwrite={overwrite}, exiting")
            return
        else:
            print(f"Computing statistic {fn_statistic} (overwrite={overwrite})")
    if len(fns_statistics) == 0:
        print(f"No statistics to compute for idx_mock={idx_mock}, exiting")
        return

    start_tot = time.time()
    
    n_grid_orig = 512 # we just know this from how we created the fields
    print(f"n_grid_orig = {n_grid_orig}", flush=True)

    # get cosmology (need for all but bispec, just get here)
    param_dict = param_dict_fixed.copy()
    if params_df is not None:
        param_dict.update(params_df.loc[idx_mock].to_dict())
    print(param_dict, flush=True)
    cosmo = utils.get_cosmo(param_dict)

    if '2Gpc' in fn_fields:  # Extract box size from file name
        box_size = 2000.0
    else:
        box_size = 1000.0

    for i, fn_statistic in enumerate(fns_statistics):
        if idxs_bias is not None:
            idx_bias = idxs_bias[i]

        if statistic == 'pnn':
            start = time.time()
            
            bias_terms_eul = get_bias_fields(fn_fields)
            power_all_terms = compute_pnn_from_bias_fields(bias_terms_eul, cosmo, box_size, n_grid_orig,
                                                    n_threads=n_threads)
            np.save(fn_statistic, power_all_terms)
            end = time.time()
            print(f"Computed {statistic} for idx_mock={idx_mock} ({fn_statistic}) in time {end-start:.2f} s", flush=True)
            
        elif statistic == 'pk':
            start = time.time()

            tracer_field = make_tracer_field(fn_fields, idx_bias, biasparams_df,
                                    biasparams_dict_fixed, Anoise_df, Anoise_dict_fixed, tag_noise)
            pk_obj = compute_pk(tracer_field, cosmo, box_size,
                                    n_threads=n_threads)
            np.save(fn_statistic, pk_obj)
            end = time.time()
            print(f"Computed {statistic} for idx_mock={idx_mock} ({fn_statistic}) in time {end-start:.2f} s", flush=True)

        elif statistic == 'pklin':
            # note: doesn't use the muchisimocks data! just takes the seed and 
            # makes a linear sim w bacco, then computes its pk
            start = time.time()
            pk_obj = compute_pk_linear(idx_mock, cosmo, box_size, n_grid_orig,
                                    n_threads=n_threads)
            np.save(fn_statistic, pk_obj)
            end = time.time()
            print(f"Computed {statistic} for idx_mock={idx_mock} ({fn_statistic}) in time {end-start:.2f} s", flush=True)

        elif statistic == 'bispec':
                            
            start = time.time()

            assert base_bispec is not None, "base_bispec must be provided for bispectrum computation"
            tracer_field = make_tracer_field(fn_fields, idx_bias, biasparams_df,
                                    biasparams_dict_fixed, Anoise_df, Anoise_dict_fixed, tag_noise)
            bspec, bk_corr = compute_bispectrum(base_bispec, tracer_field)
            k123 = bspec.get_ks()
            weight = k123.prod(axis=0)
            bispec_results_dict = {
                'k123': k123,
                'bispectrum': bk_corr,
                'weight': weight,
            }

            np.save(fn_statistic, bispec_results_dict)
        
            end = time.time()
            print(f"Computed {statistic} for idx_mock={idx_mock}, idx_bias={idx_bias} ({fn_statistic}) in time {end-start:.2f} s", flush=True)
            
    end_tot = time.time()
    print(f"Total time to compute {statistic}(s) for idx_mock={idx_mock} in time {end_tot-start_tot:.2f} s", flush=True)


def get_fns_to_compute(statistic, idx_mock, tag_mocks, tag_biasparams, tag_Anoise,
                       params_df, biasparams_df):
    # Checking here if we have remaining statistics to compute and if we need to load the fields at all
    # (because loading fields takes some time, so want to check first if we'll need them;
    # we'll also check later for the bispec case if we need to compute that particular one)
    # "precompute" means we'll add up all the fields first and compute stat on final field;
    # alternative is e.g. computing the pnn so we just need the individual fields
    
    # tag_noise is within tag_mocks
    
    # actual mock data dir is just cosmology-dependent
    dir_statistics = f'/scratch/kstoreyf/muchisimocks/data/{statistic}s_mlib/{statistic}s{tag_mocks}'
    Path.mkdir(Path(dir_statistics), parents=True, exist_ok=True)
    
    fns_statistics = []
    # check whether will need precompute later, or can just define in this func
    if tag_biasparams is not None and 'p0' not in tag_biasparams:                
        #if tag_Anoise is not None and tag_params is None:
            # compute the statistic of only the noise field!
            
        # figure out which bias indices to use
        if 'fisher' in tag_biasparams:
            idxs_bias = data_loader.get_bias_indices_for_idx(idx_mock, modecosmo='fisher',
                                                params_df=params_df, biasparams_df=biasparams_df)
        else:
            factor, longer_df = data_loader.check_df_lengths(params_df, biasparams_df)
            assert longer_df == 'bias' or longer_df == 'same', "In non-precomputed mode, biasparams_df should be longer or same length as params_df"
            idxs_bias = data_loader.get_bias_indices_for_idx(idx_mock, modecosmo='lh', factor=factor)
        
        #exist_all = True
        for idx_bias in idxs_bias:
            if tag_Anoise is not None and 'p0' not in tag_Anoise:
                # noise field associated w the bias params
                # in the case where 1 bias param per cosmo, idx_mock==idx_bias==idx_noise
                idx_noise = idx_bias
                fn_statistic = f'{dir_statistics}/{statistic}_{idx_mock}_b{idx_bias}_n{idx_noise}.npy' 
            else:
                fn_statistic = f'{dir_statistics}/{statistic}_{idx_mock}_b{idx_bias}.npy'
            fns_statistics.append(fn_statistic)
    else:
        #idxs_bias = None # bias not needed, either bc pnn, or noise-only
        idxs_bias = [0] # because still may need for noise
        # TODO deal with noise-only case
        fn_statistic = f'{dir_statistics}/{statistic}_{idx_mock}.npy'
        fns_statistics.append(fn_statistic)
    return fns_statistics, idxs_bias


def get_bias_fields(fn_fields):
    # Load the bias fields from the file
    try:
        bias_terms_eul = np.load(fn_fields)
    except FileNotFoundError:
        print(f"File {fn_fields} not found, exiting")
        return None

    return bias_terms_eul


def make_tracer_field(fn_fields, idx_bias, biasparams_df, biasparams_dict_fixed, 
                      Anoise_df, Anoise_dict_fixed, tag_noise):
    
    # load eulerian bias fields
    bias_terms_eul = get_bias_fields(fn_fields)
    
    # get bias vector
    biasparam_dict = biasparams_dict_fixed.copy()
    if biasparams_df is not None:
        biasparam_dict.update(biasparams_df.loc[idx_bias].to_dict())
    bias_vector = [biasparam_dict[name] for name in utils.biasparam_names_ordered]
    
    # TODO figure out if should be indexing bias, or cosmo, or indep...
    if tag_noise is not None:
        # get noise field
        fn_noise = f'/scratch/kstoreyf/muchisimocks/data/noise_fields/fields{tag_noise}/noise_field_n{idx_bias}.npy'
        if not os.path.exists(fn_noise):
            raise ValueError(f"Noise field {fn_noise} does not exist!")
        noise_field = np.load(fn_noise)
        
        # get A_noise
        A_noise_dict = Anoise_dict_fixed.copy()
        if Anoise_df is not None:
            assert len(Anoise_df)==len(biasparams_df), "Anoise_df should have same length as biasparams_df"
            A_noise_dict.update(Anoise_df.loc[idx_bias].to_dict())
        A_noise = A_noise_dict['A_noise']      
    else:
        noise_field = None
        A_noise = None

    n_grid_orig = 512  # we just know this from how we created the fields
    tracer_field = utils.get_tracer_field(bias_terms_eul, bias_vector,
                                            noise_field=noise_field, A_noise=A_noise, n_grid_norm=n_grid_orig)
    return tracer_field


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


def compute_pk(tracer_field, cosmo, box_size,
               log_binning=True,
               normalise_grid=False, deconvolve_grid=False,
               interlacing=False, deposit_method='cic',
               correct_grid=False,
               n_threads=8, fn_pk=None):


    k_min = 0.01
    k_max = 0.4
    n_bins = 30
    
    # NOTE by default assumes tracer field is already normalized!

    # n_grid has to match the tracer field size for this compuation!
    n_grid = tracer_field.shape[-1]
    print("Computing pk, using n_grid = ", n_grid, flush=True)

    # defaults from bacco.statistics.compute_crossspectrum_twogrids
    # unless passed or otherwise denoted
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
        "pk_lt": None,
        "kmin": k_min,
        "kmax": k_max,
        "nbins": n_bins,
        "correct_grid": correct_grid,
        "zspace": False,
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

    pk_obj = bacco.statistics.compute_crossspectrum_twogrids(
                        grid1=tracer_field,
                        grid2=tracer_field,
                        **args_power_grid)
    if fn_pk is not None:
        np.save(fn_pk, pk_obj)
    return pk_obj


def compute_pk_linear(seed, cosmo, box_size, n_grid_orig,
                        n_threads=1):
    
    expfactor = 1.0
    print("Generating ZA sim", flush=True)
    sim, disp_field = bacco.utils.create_lpt_simulation(cosmo, box_size, Nmesh=n_grid_orig, Seed=seed,
                                                        FixedInitialAmplitude=False, InitialPhase=0, 
                                                        expfactor=expfactor, LPT_order=1, order_by_order=None,
                                                        phase_type=1, ngenic_phases=True, return_disp=True, 
                                                        sphere_mode=0)
    field_lin = sim.linear_field[0]
    pk_obj = compute_pk(field_lin, cosmo, box_size, n_threads=n_threads)

    return pk_obj


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