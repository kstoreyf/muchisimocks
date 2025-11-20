from fileinput import filename
import numpy as np
import pandas as pd
from pathlib import Path
import re

import bacco

import utils


def main():
    
    #tag_noise = '_noise_quijote_p0_n1000'
    tag_noise = '_noise_p5_n10000_advected'
    #tag_noise = '_noise_test_p5_n1000'
    overwrite = False

    pattern = r'n(\d+)'
    match = re.search(pattern, tag_noise)

    if match:
        n_fields = int(match.group(1))
    else:
        raise ValueError(f"Pattern '{pattern}' not found in tag '{tag_noise}'")
    
    dir_noise = f'/scratch/kstoreyf/muchisimocks/data/noise_fields/fields{tag_noise}'
    Path(dir_noise).mkdir(parents=True, exist_ok=True)

    #make_noise_fields(n_fields, dir_noise, overwrite=overwrite)
    n_fields = 1
    make_noise_fields_advected(n_fields, dir_noise, overwrite=overwrite)


def make_noise_fields(n_fields, dir_noise, overwrite=False):


    box_size = 1000.0
    n_grid = 128
    # nbar from ELGs estimate in http://arxiv.org/abs/2307.09134, Pellejero-Ibanez et al. 2023
    # "Hybrid-bias and displacement emulators for field-level modelling of galaxy clustering in real and redshift space"
    # (ELGs: 2.3e-4)
    nbar = 2.2e-4 
    
    cell_size = box_size / n_grid  # Mpc/h per cell
    cell_volume = cell_size**3
    rms_cell = 1 / np.sqrt(nbar * cell_volume)
    
    rng = np.random.default_rng(seed=42)
    for i in range(n_fields):
        
        if i % 100 == 0:
            print(f"Generating noise field {i}/{n_fields}")
        noise_field = rms_cell * rng.standard_normal((n_grid, n_grid, n_grid))
        noise_field /= n_grid**3
        
        # Save the noise field to a file
        fn_noise = f"{dir_noise}/noise_field_n{i}.npy"
        if Path(fn_noise).exists() and not overwrite:
            #print(f"File {fn_noise} already exists. Skipping...")
            continue
        else:
            np.save(fn_noise, noise_field)
            #print(f"Saved noise field {i}/{n_fields} to {fn_noise}")



def make_noise_fields_advected(n_fields, dir_noise, overwrite=False):
    
    dir_mocks = '/scratch/kstoreyf/muchisimocks/muchisimocks_lib_p5_n10000_rerun'
    subdir_prefix = 'LH'
    idx_mock = 0
    dir_LH = f'{dir_mocks}/{subdir_prefix}{idx_mock}'
    fn_disp = f'{dir_LH}/pred_disp.npy'
    pred_disp = np.load(fn_disp, allow_pickle=True)
    
    n_grid_orig = 512
    n_grid_target = 128
    box_size = 1000.0
    nbar_fid = 0.00022
    
    rng = np.random.default_rng(seed=42)
    for i in range(n_fields):
        
        if i % 100 == 0:
            print(f"Generating noise field {i}/{n_fields}")
            
        noise_field_hires = gen_noise_field(nbar_fid, n_grid_orig, box_size, seed=i)
            
        ## Create regular grid and displace particles
        print("Generating grid", flush=True)
        grid = bacco.visualization.uniform_grid(npix=n_grid_orig, L=box_size, ndim=3, bounds=False)

        print(pred_disp.shape)
        print(grid.shape)
        print((grid.reshape(-1,3)).shape)
        print(box_size)
        print("Adding predicted displacements", flush=True)
        pred_pos = bacco.scaler.add_displacement(None,
                                        pred_disp,
                                        box=box_size,
                                        pos=grid.reshape(-1,3),
                                        vel=None,
                                        vel_factor=0,
                                        verbose=True)[0]

        ### Advect noise field to Eulerian space
        print(noise_field_hires.shape)
        print((noise_field_hires).flatten().shape)
        print(n_grid_orig)
        noise_field_eul_hires = bacco.statistics.compute_mesh(ngrid=n_grid_orig, box=box_size, pos=pred_pos, 
                                        mass = (noise_field_hires).flatten(), deposit_method='cic', 
                                        interlacing=False)
        noise_field_eul_hires = np.squeeze(noise_field_eul_hires)

        noise_field_eul = utils.remove_highk_modes(noise_field_eul_hires, box_size, n_grid_target)

        # Save the noise field to a file
        fn_noise = f"{dir_noise}/noise_field_n{i}.npy"
        if Path(fn_noise).exists() and not overwrite:
            #print(f"File {fn_noise} already exists. Skipping...")
            continue
        else:
            np.save(fn_noise, noise_field_eul)
            #print(f"Saved noise field {i}/{n_fields} to {fn_noise}")



def gen_noise_field(nbar_fid, n_grid, box_size, seed):
    cell_size = box_size / n_grid  # Mpc/h per cell
    cell_volume = cell_size**3
    rms_cell = 1 / np.sqrt(nbar_fid * cell_volume)

    rng = np.random.default_rng(seed=seed)

    noise_field = rms_cell * rng.standard_normal((n_grid, n_grid, n_grid))
    noise_field /= n_grid**3
    return noise_field


if __name__ == "__main__":
    main()