import os
import numpy as np
import itertools
import time
import argparse
from pathlib import Path

import bacco
#import PolyBin3D as pb

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
    parser.add_argument('--tag_params', type=str, default=None,
                        help='Tag for parameter set')
    parser.add_argument('--tag_biasparams', type=str, default=None,
                        help='Tag for bias parameter set')
    parser.add_argument('--tag_noise', type=str, default=None,
                        help='Tag for noise fields')
    parser.add_argument('--tag_Anoise', type=str, default=None,
                        help='Tag for noise parameter set')
    parser.add_argument('--idx_mock_start', type=int,
                        help='Index of the LH realization to start')
    parser.add_argument('--idx_mock_end', type=int,
                        help='Index of the LH realization to end')
    parser.add_argument('--n_threads', type=int, default=1,
                        help='Number of threads for calculation')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing files')
    args = parser.parse_args()

    statistic = args.statistic
    tag_params = args.tag_params
    tag_biasparams = args.tag_biasparams
    tag_noise = args.tag_noise
    tag_Anoise = args.tag_Anoise
    n_threads = args.n_threads
    overwrite = args.overwrite

    stats_supported = ['pnn', 'pk', 'pklin', 'bispec', 'pgm']
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
        #print(idx_mock)
        run(statistic, idx_mock,
            tag_params=tag_params, tag_biasparams=tag_biasparams,
            tag_noise=tag_noise, tag_Anoise=tag_Anoise,
            n_threads=n_threads, overwrite=overwrite, 
            base_bispec=base)
        # try:
        #     run(statistic, idx_mock,
        #     tag_params=tag_params, tag_biasparams=tag_biasparams,
        #     tag_noise=tag_noise, tag_Anoise=tag_Anoise,
        #     n_threads=n_threads, overwrite=overwrite, 
        #     base_bispec=base)
        # except Exception as e:
        #     print(f"Failed to compute statistic for idx_mock={idx_mock}: {e}")
        #     print("Continuing to next index...")
        #     continue
        
        

def run_loop():
    ## main training set
    #n_mocks = 10000
    #tag_params = f'_p5_n{n_mocks}'    
    #tag_biasparams = '_biaszen_p4_n10000'

    ## fixed cosmo test set
    # n_mocks = 1000
    # tag_params = f'_quijote_p0_n{n_mocks}'
    # tag_biasparams = '_b1000_p0_n1' # for pnn, don't need biasparams
    # #tag_biasparams = None # for pnn, don't need biasparams
    # tag_noise = '_noise_quijote_p0_n1000'
    # tag_Anoise = '_An1_p0_n1'
    ## variable cosmo test set
    #n_mocks = 1000
    #tag_params = f'_test_p5_n{n_mocks}'

    ## fisher
    #tag_params = '_fisher_quijote'
    #tag_biasparams = None # for pnn, don't need biasparams
    #tag_biasparams = '_fisher_biaszen'
    #n_mocks = 21 #magic, i just know this number for fisher; TODO read it in
    
    ## noise only
    n_mocks = 1000
    tag_params = None # for noise-only, no params
    tag_biasparams = None # for noise-only, no bias params
    tag_noise = '_noise_quijote_p0_n1000'
    tag_Anoise = None # will be computing stats directly of noise field, no A_noise needed
    
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
        run(statistic, idx_mock, 
            tag_params=tag_params,
            tag_biasparams=tag_biasparams,
            tag_noise=tag_noise, tag_Anoise=tag_Anoise,
            base_bispec=base,
            n_threads=n_threads, overwrite=overwrite, 
            )    
    

def setup_bispsec(box_size, n_grid, n_threads):
    # Load the PolyBin3D class 
    import PolyBin3D as pb
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
    
    
## updated version by claude
def run(statistic, idx_mock,
        tag_params=None, tag_biasparams=None, tag_noise=None, tag_Anoise=None, 
        base_bispec=None, n_threads=1, overwrite=False):

    print(f"Starting muchisimocks {statistic} computation for idx_mock={idx_mock}", flush=True)

    params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, Anoise_df, Anoise_dict_fixed, _, _ = \
        data_loader.load_params(tag_params, tag_biasparams, tag_Anoise)
    print('tag_Anoise =',tag_Anoise)
    print(len(Anoise_df) if Anoise_df is not None else "Anoise_df is None", flush=True)

    if tag_params is not None:  
        if 'p0' in tag_params or 'fisher' in tag_params:
            subdir_prefix = 'mock'
        else:
            subdir_prefix = 'LH'
        # this would only not be true in the case of noise-only computation
        dir_mocks = f'/scratch/kstoreyf/muchisimocks/muchisimocks_lib{tag_params}'
        fn_fields = f'{dir_mocks}/{subdir_prefix}{idx_mock}/bias_fields_eul_deconvolved_{idx_mock}.npy'
        
        # get cosmology (need for all but bispec, just get here)
        param_dict = param_dict_fixed.copy()
        if params_df is not None:
            param_dict.update(params_df.loc[idx_mock].to_dict())
        print(param_dict, flush=True)
        cosmo = utils.get_cosmo(param_dict)
    else:
        # noise-only computation, no fields to load
        fn_fields = None
        # pk needs a cosmo object, but in noise case we dont have a cosmo...
        # docs say can pass None but it breaks things
        # TODO this will be an issue for future tests!
        # for now we'll load a fiducial
        cosmo = utils.get_cosmo(utils.cosmo_dict_quijote)
    
    # Use the shared data_loader functions for consistent logic
    # Special handling for pnn - it only needs one file per cosmology regardless of bias parameters
    if statistic == 'pnn':
        dir_statistics = data_loader.get_dir_statistics('pnn', tag_params, None)
        fn_statistic = f'{dir_statistics}/pnn_{idx_mock}.npy'
        fns_statistics_all = [fn_statistic]
        idxs_bias = [None]
        idxs_noise = [None]
    else:
        # For other statistics, use the shared function
        dir_statistics, fns_statistics_all, idxs_bias, idxs_noise = data_loader.get_fns_statistic(
            statistic, idx_mock, tag_params, tag_biasparams, tag_noise, tag_Anoise,
            params_df, biasparams_df
        )
        
    # Create directory if it doesn't exist
    os.makedirs(dir_statistics, exist_ok=True)
    
    # Filter out files that already exist (unless overwriting)
    files_to_compute = []
    idxs_bias_to_compute = []
    idxs_noise_to_compute = []
    
    for i, fn_stat in enumerate(fns_statistics_all):
        if os.path.exists(fn_stat) and not overwrite:
            print(f"Statistic {fn_stat} exists and overwrite={overwrite}, not recomputing")
        else:
            files_to_compute.append(fn_stat)
            if idxs_bias is not None:
                idxs_bias_to_compute.append(idxs_bias[i])
            else:
                idxs_bias_to_compute.append(None)
            if idxs_noise is not None:
                idxs_noise_to_compute.append(idxs_noise[i])
            else:
                idxs_noise_to_compute.append(None)
    
    if len(files_to_compute) == 0:
        print(f"No statistics to compute for idx_mock={idx_mock}, exiting")
        return

    start_tot = time.time()
    box_size = 1000.0
    n_grid_orig = 512 # we just know this from how we created the fields

    # Main computation loop
    for i, fn_statistic in enumerate(files_to_compute):
        
        try: 
            print(f"Computing {statistic} for idx_mock={idx_mock}, fn_statistic={fn_statistic}", flush=True)
            
            idx_bias = idxs_bias_to_compute[i]
            idx_noise = idxs_noise_to_compute[i]

            if statistic == 'pnn':
                start = time.time()
                
                bias_terms_eul = get_bias_fields(fn_fields)
                power_all_terms = compute_pnn_from_bias_fields(bias_terms_eul, cosmo, box_size, n_grid_orig,
                                                        n_threads=n_threads, fn_stat=fn_statistic)
                end = time.time()
                print(f"Computed {statistic} for idx_mock={idx_mock} ({fn_statistic}) in time {end-start:.2f} s", flush=True)
                
            elif statistic == 'pk':
                start = time.time()

                tracer_field = make_tracer_field(fn_fields, idx_bias, idx_noise, tag_noise, biasparams_df,
                                        biasparams_dict_fixed, Anoise_df, Anoise_dict_fixed,
                                        n_grid_orig)
                compute_pk(tracer_field, cosmo, box_size,
                                        n_threads=n_threads, fn_stat=fn_statistic)
                end = time.time()
                print(f"Computed {statistic} for idx_mock={idx_mock} ({fn_statistic}) in time {end-start:.2f} s", flush=True)

            elif statistic == 'pklin':
                # note: doesn't use the muchisimocks data! just takes the seed and 
                # makes a linear sim w bacco, then computes its pk
                start = time.time()
                compute_pk_linear(idx_mock, cosmo, box_size, n_grid_orig,
                                        n_threads=n_threads, fn_stat=fn_statistic)
                end = time.time()
                print(f"Computed {statistic} for idx_mock={idx_mock} ({fn_statistic}) in time {end-start:.2f} s", flush=True)

            elif statistic == 'pgm':
                start = time.time()

                tracer_field = make_tracer_field(fn_fields, idx_bias, idx_noise, tag_noise, biasparams_df,
                                        biasparams_dict_fixed, Anoise_df, Anoise_dict_fixed,
                                        n_grid_orig)
                bias_terms_eul = get_bias_fields(fn_fields)
                # Get the second bias field (index 1)
                matter_density_field = bias_terms_eul[1]
                # normalize the matter density field by n_grid_orig**3 for muchisimocks
                matter_density_field_norm = matter_density_field/n_grid_orig**3
                compute_pgm(tracer_field, matter_density_field_norm, cosmo, box_size,
                                        n_threads=n_threads, fn_stat=fn_statistic)
                end = time.time()
                print(f"Computed {statistic} for idx_mock={idx_mock} ({fn_statistic}) in time {end-start:.2f} s", flush=True)

            elif statistic == 'bispec':
                                
                start = time.time()

                assert base_bispec is not None, "base_bispec must be provided for bispectrum computation"
                tracer_field = make_tracer_field(fn_fields, idx_bias, idx_noise, tag_noise, biasparams_df,
                                        biasparams_dict_fixed, Anoise_df, Anoise_dict_fixed,
                                        n_grid_orig)
                compute_bispectrum(base_bispec, tracer_field, fn_stat=fn_statistic)
            
                end = time.time()
                print(f"Computed {statistic} for idx_mock={idx_mock}, idx_bias={idx_bias} ({fn_statistic}) in time {end-start:.2f} s", flush=True)
        except Exception as e:
            raise e
            #print(f"Failed to compute {statistic} for idx_mock={idx_mock}, fn_statistic={fn_statistic}: {e}")
            #print("cleaning up and continuing to next statistic...")
            #Path(fn_statistic).unlink(missing_ok=True)
            #continue
            
    end_tot = time.time()
    print(f"Total time to compute {statistic}(s) for idx_mock={idx_mock} in time {end_tot-start_tot:.2f} s", flush=True)


def get_bias_fields(fn_fields):
    # Load the bias fields from the file
    print("Getting bias fields")
    print(fn_fields)
    try:
        bias_terms_eul = np.load(fn_fields)
        print(bias_terms_eul.shape)
    except FileNotFoundError:
        print(f"File {fn_fields} not found, exiting")
        return None

    return bias_terms_eul


def make_tracer_field(fn_fields, idx_bias, idx_noise, tag_noise, biasparams_df, biasparams_dict_fixed, 
                      Anoise_df, Anoise_dict_fixed, n_grid_orig):
    
    print("make_tracer_field")
    print(fn_fields)
    print(idx_bias, idx_noise)
    print(tag_noise)
    
    if tag_noise is not None:
        # Check if idx_noise is None when we need it
        if idx_noise is None:
            raise ValueError(f"idx_noise is None but tag_noise='{tag_noise}' was provided. "
                           f"This suggests a mismatch in the logic for determining noise indices. "
                           f"Check that tag_Anoise and other noise-related parameters are correctly set.")
        
        # get noise field
        fn_noise = f'/scratch/kstoreyf/muchisimocks/data/noise_fields/fields{tag_noise}/noise_field_n{idx_noise}.npy'
        if not os.path.exists(fn_noise):
            raise ValueError(f"Noise field {fn_noise} does not exist!")
        noise_field = np.load(fn_noise)
        
        # get A_noise; can have noise without Anoise, but if have Anoise will also need noise
        # noise-field computation only
        if Anoise_df is None and Anoise_dict_fixed is None:
            A_noise = None
        else:
            # when one of these is not None, this should get us A_noise for fixed or varying
            A_noise_dict = Anoise_dict_fixed.copy()
            if Anoise_df is not None:
                A_noise_dict.update(Anoise_df.loc[idx_noise].to_dict())
            # A_noise is now a vector
            A_noise = [A_noise_dict[npm] for npm in utils.noiseparam_names_ordered]  
    else:
        noise_field = None
        A_noise = None

    if fn_fields is not None:
        # load eulerian bias fields
        bias_terms_eul = get_bias_fields(fn_fields)
        # get bias vector
        biasparam_dict = biasparams_dict_fixed.copy()
        if biasparams_df is not None:
            biasparam_dict.update(biasparams_df.loc[idx_bias].to_dict())
        bias_vector = [biasparam_dict[name] for name in utils.biasparam_names_ordered]
        # make tracer field
        tracer_field = utils.get_tracer_field(bias_terms_eul, bias_vector, n_grid_orig,
                                                noise_field=noise_field, A_noise=A_noise)
    else:
        # noise-only
        tracer_field = noise_field
        
    return tracer_field


def compute_pnn_from_bias_fields(bias_terms_eul, cosmo, box_size, n_grid_orig,
                        n_threads=1, fn_stat=None):

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

    if fn_stat is not None:
        Path.absolute(Path(fn_stat).parent).mkdir(parents=True, exist_ok=True)
        np.save(fn_stat, power_all_terms)
        
    return power_all_terms


def compute_pk(tracer_field, cosmo, box_size,
               k_min=0.01, k_max=0.4, n_bins=30,
               log_binning=True,
               normalise_grid=False, deconvolve_grid=False,
               interlacing=False, deposit_method='cic',
               correct_grid=False,
               n_threads=8, fn_stat=None):
    
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
    
    if fn_stat is not None:
        Path.absolute(Path(fn_stat).parent).mkdir(parents=True, exist_ok=True)
        np.save(fn_stat, pk_obj)
        
    return pk_obj


def compute_pk_linear(seed, cosmo, box_size, n_grid_orig,
                        n_threads=1, fn_stat=None):
    
    expfactor = 1.0
    print("Generating ZA sim", flush=True)
    sim, disp_field = bacco.utils.create_lpt_simulation(cosmo, box_size, Nmesh=n_grid_orig, Seed=seed,
                                                        FixedInitialAmplitude=False, InitialPhase=0, 
                                                        expfactor=expfactor, LPT_order=1, order_by_order=None,
                                                        phase_type=1, ngenic_phases=True, return_disp=True, 
                                                        sphere_mode=0)
    field_lin = sim.linear_field[0]
    pk_obj = compute_pk(field_lin, cosmo, box_size, n_threads=n_threads, fn_stat=fn_stat)
        
    return pk_obj


def compute_pgm(tracer_field, matter_density_field, cosmo, box_size,
                k_min=0.01, k_max=0.4, n_bins=30,
               log_binning=True,
               normalise_grid=False, deconvolve_grid=False,
               interlacing=False, deposit_method='cic',
               correct_grid=False,
               n_threads=8, fn_stat=None):
    
    # make sure to pre-normalize the matter density field!
    # for muchisimocks, divide by n_grid_orig**3; for SHAMe, divide by np.sum(matter_density_field) (??)

    # n_grid has to match the tracer field size for this computation!
    n_grid = tracer_field.shape[-1]
    print("Computing pgm, using n_grid = ", n_grid, flush=True)

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

    pgm_obj = bacco.statistics.compute_crossspectrum_twogrids(
                        grid1=tracer_field,
                        grid2=matter_density_field,
                        **args_power_grid)
    
    if fn_stat is not None:
        Path.absolute(Path(fn_stat).parent).mkdir(parents=True, exist_ok=True)
        np.save(fn_stat, pgm_obj)
        
    return pgm_obj


def compute_bispectrum(base, tracer_field, k_min=0.01, k_max=0.4, n_bins=7, fn_stat=None):

    k_edges = np.linspace(k_min, k_max, n_bins+1) # 7 bins
    k_edges_squeeze = k_edges.copy()
    lmax = 1
    
    # Load the bispectrum class
    import PolyBin3D as pb
    bspec = pb.BSpec(base, 
                    k_edges, # one-dimensional bin edges
                    applySinv = None, # weighting function [only needed for unwindowed estimators]
                    mask = None, # real-space mask
                    lmax = lmax, # maximum Legendre multipole
                    k_bins_squeeze = k_edges_squeeze, # squeezed bins
                    include_partial_triangles = False, # whether to include bins whose centers do not satisfy triangle conditions
                    )
    
    bk_corr = bspec.Bk_ideal(tracer_field, discreteness_correction=True)

    if fn_stat is not None:
        save_bispectrum(fn_stat, bspec, bk_corr)
            
    return bspec, bk_corr


def save_bispectrum(fn_stat, bspec, bk_corr, n_grid=None):
    k123 = bspec.get_ks()
    weight = k123.prod(axis=0)
    bispec_results_dict = {
        'k123': k123,
        'bispectrum': bk_corr,
        'weight': weight,
    }
    if n_grid is not None:
        bispec_results_dict['n_grid'] = n_grid
    Path.absolute(Path(fn_stat).parent).mkdir(parents=True, exist_ok=True)
    np.save(fn_stat, bispec_results_dict)
    return


if __name__ == '__main__':
    main()