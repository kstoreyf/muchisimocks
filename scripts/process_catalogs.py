"""
Process SHAMe catalogs for comparison with muchisimocks data.

This script processes galaxy catalogs from SHAMe mocks and computes
power spectra and bispectra for comparison with muchisimocks training/test data.
"""

import h5py
import numpy as np
from pathlib import Path
import pyfftw

import bacco
import bacco.probabilistic_bias as pb

import sys
sys.path.append('/dipc/kstoreyf/muchisimocks/scripts')
import utils
import compute_statistics as cs



def main():
    """
    Main function to process SHAMe catalog and compare with muchisimocks data.
    """
    # Configuration
    tag_mock = '_nbar0.00011'  
    #tag_mock = '_nbar0.00022'  
    #tag_mock = '_nbar0.00054'  
    #tag_mock = '_An1_deconvolve'
    #tag_mock = '_An1_orig'
    if tag_mock=='_An1' or tag_mock=='_An1_deconvolve':
        dir_cat = '../data/shame_catalogues_to_share/kate'
        fn_cat0 = f'{dir_cat}/kate_sham_catalogue_a1.0_par_b_Planck_N3072_L1024_0.00_0.00022.h5' #_An1
        fn_catpi = f'{dir_cat}/kate_sham_catalogue_a1.0_par_b_Planck_N3072_L1024_3.14_0.00022.h5'
    elif tag_mock=='_An1_orig':
        dir_cat = '../data/shame_catalogues_to_share' 
        fn_cat0 = f'{dir_cat}/kate_sham_catalogue_a1.0_par_b_Planck_N3072_L1024_0.00.h5' #'_An1_orig
        fn_catpi = f'{dir_cat}/kate_sham_catalogue_a1.0_par_b_Planck_N3072_L1024_3.14.h5'
    elif tag_mock.startswith('_nbar'):
        nbar = tag_mock.split('nbar')[-1]
        dir_cat = '../data/shame_catalogues_to_share/kate'
        fn_cat0 = f'{dir_cat}/kate_sham_catalogue_a1.0_par_b_Planck_N3072_L1024_0.00_{nbar}.h5'
        fn_catpi = f'{dir_cat}/kate_sham_catalogue_a1.0_par_b_Planck_N3072_L1024_3.14_{nbar}.h5'
    else:
        raise ValueError(f"tag_mock {tag_mock} not recognized!")
    
    data_mode = 'shame'
    statistics = ['pk', 'bispec']
    overwrite = True
    
    save_indiv_phases = True  # Whether to save individual phase statistics
    
    # Grid and box parameters
    box_size_mock = 1024.0
    
    print("=== Processing SHAMe Catalog ===")
    
    # Process catalog to tracer_field field
    fn_cat0_mesh = f'../data/data_{data_mode}/data{tag_mock}/tracer_field_phase0.npy'
    tracer_field_0, n_grid_mock_0 = process_catalog_to_mesh(
        fn_cat0, box_size_mock, fn_cat_mesh=fn_cat0_mesh, overwrite=overwrite
    )
    fn_catpi_mesh = f'../data/data_{data_mode}/data{tag_mock}/tracer_field_phasepi.npy'
    tracer_field_pi, n_grid_mock_pi = process_catalog_to_mesh(
        fn_catpi, box_size_mock, fn_cat_mesh=fn_catpi_mesh, overwrite=overwrite
    )
    
    # ok to assume we know cosmo bc just use for periodic box correction
    cosmo = utils.get_cosmo(utils.cosmo_dict_shame)
    #cosmo = None #feel like should be assuming i don't know cosmo?

    for statistic in statistics:
        dir_statistics = f'../data/data_{data_mode}/data{tag_mock}/{statistic}s'
        dir_statistics_phase0 = f'../data/data_{data_mode}/data{tag_mock}_phase0/{statistic}s'
        dir_statistics_phasepi = f'../data/data_{data_mode}/data{tag_mock}_phasepi/{statistic}s'
        fn_stat = f'{dir_statistics}/{statistic}.npy'
        if not overwrite and Path(fn_stat).exists():
            print(f"Statistic {statistic} already computed at {fn_stat}, skipping")
            continue
        
        if statistic=='pk':
            print("\n=== Computing Power Spectrum ===")
            pk_0 = compute_pk(tracer_field_0, box_size_mock, cosmo=cosmo)
            pk_pi = compute_pk(tracer_field_pi, box_size_mock, cosmo=cosmo)
            if save_indiv_phases:
                fn_stat_0 = f'{dir_statistics_phase0}/{statistic}.npy'
                fn_stat_pi = f'{dir_statistics_phasepi}/{statistic}.npy'
                Path.absolute(Path(fn_stat_0).parent).mkdir(parents=True, exist_ok=True)
                np.save(fn_stat_0, pk_0)
                print(f"Power spectrum for phase 0 saved to {fn_stat_0}")
                Path.absolute(Path(fn_stat_pi).parent).mkdir(parents=True, exist_ok=True)
                np.save(fn_stat_pi, pk_pi)
                print(f"Power spectrum for phase pi saved to {fn_stat_pi}")
            pk_mean = {} #for now ignoring any other entries in the dict, haven't been using
            pk_mean['k'] = pk_0['k'] 
            pk_mean['pk'] = 0.5 * (pk_0['pk'] + pk_pi['pk'])
            pk_mean['pk_gaussian_error'] = 0.5 * (pk_0['pk_gaussian_error'] + pk_pi['pk_gaussian_error']) #??
            Path.absolute(Path(fn_stat).parent).mkdir(parents=True, exist_ok=True)
            np.save(fn_stat, pk_mean)
            print(f"Power spectrum saved to {fn_stat}")
        
        elif statistic=='bispec':
            print("\n=== Computing Bispectrum ===")
            bspec_0, bk_corr_0 = compute_bispectrum(
                tracer_field_0, box_size_mock, n_grid_mock_0)
            bspec_pi, bk_corr_pi = compute_bispectrum(
               tracer_field_pi, box_size_mock, n_grid_mock_pi)
            if save_indiv_phases:
                fn_stat_0 = f'{dir_statistics_phase0}/{statistic}.npy'
                fn_stat_pi = f'{dir_statistics_phasepi}/{statistic}.npy'
                Path.absolute(Path(fn_stat_0).parent).mkdir(parents=True, exist_ok=True)
                cs.save_bispectrum(fn_stat_0, bspec_0, bk_corr_0, n_grid=n_grid_mock_0)
                print(f"Bispectrum for phase 0 saved to {fn_stat_0}")
                Path.absolute(Path(fn_stat_pi).parent).mkdir(parents=True, exist_ok=True)
                cs.save_bispectrum(fn_stat_pi, bspec_pi, bk_corr_pi, n_grid=n_grid_mock_pi)
                print(f"Bispectrum for phase pi saved to {fn_stat_pi}")
            
            bspec_mean = bspec_0  # Both should have same k-bins
            bk_corr_mean = {} #for now ignoring any other entries in the dict, haven't been using
            bk_corr_mean['b0'] = 0.5 * (bk_corr_0['b0'] + bk_corr_pi['b0'])
            cs.save_bispectrum(fn_stat, bspec_mean, bk_corr_mean, n_grid=n_grid_mock_0)
            print(f"Bispectrum saved to {fn_stat}")

        else:
            raise ValueError(f"Statistic {statistic} not recognized!")
        
    print(f"Finished processing SHAMe catalog {fn_cat0} (and its pi partner)")
    

def round_to_nearest_even(x):
    """Round a number to the nearest even integer (needed for FFTs)."""
    return int(round(x / 2) * 2)


def process_catalog_to_mesh(fn_cat, box_size_mock, fn_cat_mesh=None, 
                            n_grid_target=128, n_grid_orig=512, box_size_muchisimocks=1000.0, 
                            cosmo=None,
                            overwrite=False):
    """
    Process a catalog file to create density mesh and tracer_field field.
    
    Parameters:
    -----------
    fn_cat : str
        Path to the catalog HDF5 file
    fn_cat_mesh : str
        Path to save the intermediate mesh file
    n_grid_target : int, optional
        Target grid size for the final mesh (default: 128)
    box_size_mock : float, optional
        Box size of the mock catalog in Mpc/h (default: 1024.0)
    box_size_muchisimocks : float, optional
        Box size used in muchisimocks in Mpc/h (default: 1000.0)
        
    Returns:
    --------
    tracer_field : ndarray
        tracer_field field normalized for comparison
    """
    if not overwrite and fn_cat_mesh is not None and Path(fn_cat_mesh).exists():
        tracer_field = np.load(fn_cat_mesh, allow_pickle=True)
        n_grid_mock = tracer_field.shape[0]
        print(f"Loaded precomputed tracer field from {fn_cat_mesh}, with n_grid_mock={n_grid_mock}")
        return tracer_field, n_grid_mock

    # Load catalog
    with h5py.File(fn_cat, 'r') as cat:
        positions = cat['gal_pos'][:]
        n_galaxies = len(positions)
        
    print(f"Loaded catalog with {n_galaxies} galaxies")
    print(f"Box size appears to be: {positions.max(axis=0)} Mpc/h")
    
    # Calculate grid sizes
    n_grid_orig_mock = round_to_nearest_even(box_size_mock / (box_size_muchisimocks/n_grid_orig))
    n_grid_mock = round_to_nearest_even(box_size_mock / (box_size_muchisimocks/n_grid_target))
    
    print(f"Resampling to {n_grid_orig_mock}^3 / {n_grid_mock}^3 grid (from {n_grid_orig}^3/{n_grid_target}^3 original grid)")
    
    # Create high-resolution mesh
    cat_mesh_ngorig = bacco.statistics.compute_mesh(
        ngrid=n_grid_orig_mock, 
        box=box_size_mock, 
        pos=positions, 
        vel=None, 
        mass=None,
        interlacing=False, 
        deposit_method='cic',
        zspace=False, 
        cosmology=cosmo
    )
    print(f"High-res mesh shape: {cat_mesh_ngorig.shape}")
    
    # Remove high-k modes to downsample
    cat_field_kcut = utils.remove_highk_modes(cat_mesh_ngorig[0], box_size_mock=box_size_mock, n_grid_target=n_grid_mock)
    
    # tested in data_creation_pipeline that doing kcut then deconvolve is basically equivalent to deconvolve then kcut,
    # and much faster
    cat_field_kcut_deconvolved = pb.convolve_linear_interpolation_kernel(cat_field_kcut, 
                                                                        npix=n_grid_orig_mock, mode="deconvolve")
    
    # Convert to tracer_field
    cat_overdensity = (cat_field_kcut_deconvolved - np.mean(cat_field_kcut_deconvolved))/np.mean(cat_field_kcut_deconvolved)
    cat_overdensity /= n_grid_mock**3 
    
    if fn_cat_mesh is not None:
        Path.absolute(Path(fn_cat_mesh).parent).mkdir(parents=True, exist_ok=True)
        print(f"Saving mesh to {fn_cat_mesh}")
        np.save(fn_cat_mesh, cat_overdensity)

    return cat_overdensity, n_grid_mock


def compute_pk(tracer_field, box_size_mock, cosmo=None, fn_stat=None,):
    if cosmo is None:
        print("No cosmology provided, using Quijote cosmology for p(k) calc")
        cosmo = utils.get_cosmo(utils.cosmo_dict_quijote)
        print(f"Using cosmology: {cosmo}")
    pk_obj = cs.compute_pk(tracer_field, cosmo, box_size_mock, fn_stat=fn_stat)
    return pk_obj


def compute_bispectrum(tracer_field, box_size_mock, n_grid, fn_stat=None, n_threads=8):
    base = cs.setup_bispsec(box_size_mock, n_grid, n_threads=n_threads)
    bspec, bk_corr = cs.compute_bispectrum(base, tracer_field, fn_stat=fn_stat)
    return bspec, bk_corr


if __name__ == "__main__":
    main()
