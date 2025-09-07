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

import sys
sys.path.append('/dipc/kstoreyf/muchisimocks/scripts')
import utils
import compute_statistics as cs


def main():
    """
    Main function to process SHAMe catalog and compare with muchisimocks data.
    """
    # Configuration
    dir_cat = '../data/shame_catalogues_to_share'
    fn_cat = f'{dir_cat}/kate_sham_catalogue_a1.0_par_b_Planck_N3072_L1024_0.00.h5'
    data_mode = 'shame'
    tag_mock = '_An1'
    statistics = ['pk', 'bispec']
    overwrite = False
    
    # Grid and box parameters
    box_size_mock = 1024.0
    
    print("=== Processing SHAMe Catalog ===")
    
    # Process catalog to tracer_field field
    fn_cat_mesh = f'../data/data_{data_mode}/data{tag_mock}/tracer_field.npy'
    tracer_field, n_grid_mock = process_catalog_to_mesh(
        fn_cat, box_size_mock, fn_cat_mesh=fn_cat_mesh, overwrite=overwrite
    )
    
    for statistic in statistics:
        dir_statistics = f'../data/data_{data_mode}/data{tag_mock}/{statistic}s'
        fn_stat = f'{dir_statistics}/{statistic}.npy'
        if not overwrite and Path(fn_stat).exists():
            print(f"Statistic {statistic} already computed at {fn_stat}, skipping")
            continue
        
        if statistic=='pk':
            print("\n=== Computing Power Spectrum ===")
            compute_pk(tracer_field, box_size_mock,
                                 fn_stat=fn_stat)
            print(f"Power spectrum saved to {fn_stat}")
        
        elif statistic=='bispec':
            print("\n=== Computing Bispectrum ===")
            compute_bispectrum(
                tracer_field, box_size_mock, n_grid_mock,
                fn_stat=fn_stat
            )
            print(f"Bispectrum saved to {fn_stat}")

        else:
            raise ValueError(f"Statistic {statistic} not recognized!")
        
    print(f"Finished processing SHAMe catalog {fn_cat}")
    

def round_to_nearest_even(x):
    """Round a number to the nearest even integer (needed for FFTs)."""
    return int(round(x / 2) * 2)


# TODO unify with function in data_creation_pipeline.py
def remove_highk_modes(field, box_size_mock, n_grid_target):
    """
    Remove high-k modes from a field to downsample it to a target grid size.
    
    Parameters:
    -----------
    field : array_like
        Input field to filter
    box_size_mock : float
        Box size of the mock in Mpc/h
    n_grid_target : int
        Target grid size (must be even)
        
    Returns:
    --------
    field_kcut : ndarray
        Filtered field with high-k modes removed
    """
    n_grid = field.shape[-1]
    k_nyq = np.pi/box_size_mock*n_grid_target
    kmesh = bacco.visualization.np_get_kmesh( (n_grid, n_grid, n_grid), box_size_mock, real=True)
    mask = (kmesh[:,:,:,0]<=k_nyq) & (kmesh[:,:,:,1]<=k_nyq) & (kmesh[:,:,:,2]<=k_nyq) & (kmesh[:,:,:,0]>-k_nyq) & (kmesh[:,:,:,1]>-k_nyq) & (kmesh[:,:,:,2]>-k_nyq)
    assert n_grid_target%2==0, "n_grid_target must be even!"

    deltak = pyfftw.builders.rfftn(field, auto_align_input=False, auto_contiguous=False, avoid_copy=True)
    deltakcut = deltak()[mask]
    deltakcut= deltakcut.reshape(n_grid_target, n_grid_target, int(n_grid_target/2)+1)
    field_kcut = pyfftw.builders.irfftn(deltakcut, axes=(0,1,2))()
        
    return field_kcut


def process_catalog_to_mesh(fn_cat, box_size_mock, fn_cat_mesh=None, 
                            n_grid_target=128, n_grid_orig=512, box_size_muchisimocks=1000.0, 
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
        cosmology=None
    )
    print(f"High-res mesh shape: {cat_mesh_ngorig.shape}")
    
    # Remove high-k modes to downsample
    cat_field_kcut = remove_highk_modes(cat_mesh_ngorig[0], box_size_mock=box_size_mock, n_grid_target=n_grid_mock)
    
    # Convert to tracer_field
    cat_overdensity = (cat_field_kcut - np.mean(cat_field_kcut))/np.mean(cat_field_kcut)
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
