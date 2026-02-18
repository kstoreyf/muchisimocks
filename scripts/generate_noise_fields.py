from fileinput import filename
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import re
import time

import bacco

import utils


def main():
    
    #tag_noise = '_noise_quijote_p0_n1000'
    #tag_noise = '_noise_p5_n10000'
    #tag_noise = '_noise_p5_n10000_advected'
    #tag_params = '_p5_n10000'
    #tag_params = '_test_p5_n1000'
    #tag_params = '_quijote_p0_n1000'
    tag_params = '_shame_p0_n1000'
    #tag_noise = f'_noise_unit{tag_params}'
    tag_noise = f'_noise{tag_params}'

    n_fields = tag_params.split('_n')[-1]
    print("n_fields =", n_fields)
    
    seed_base_dict = {
        '_p5_n10000': 0,
        '_quijote_p0_n1000': 10000,
        '_test_p5_n1000': 20000,
        '_shame_p0_n1000': 30000,
    }
    seed_base = seed_base_dict[tag_params]
    
    overwrite = True

    pattern = r'n(\d+)'
    match = re.search(pattern, tag_noise)

    if match:
        n_fields = int(match.group(1))
    else:
        raise ValueError(f"Pattern '{pattern}' not found in tag '{tag_noise}'")
     
    dir_noise = f'/scratch/kstoreyf/muchisimocks/data/noise_fields/fields{tag_noise}'
    Path(dir_noise).mkdir(parents=True, exist_ok=True)
    
    make_noise_fields(n_fields, dir_noise, seed_base, overwrite=overwrite)
    #make_noise_fields_unit(n_fields, dir_noise, seed_base, overwrite=overwrite)


def make_noise_fields(n_fields, dir_noise, seed_base, overwrite=False):


    box_size = 1000.0
    n_grid = 128
    # nbar from ELGs estimate in http://arxiv.org/abs/2307.09134, Pellejero-Ibanez et al. 2023
    # "Hybrid-bias and displacement emulators for field-level modelling of galaxy clustering in real and redshift space"
    nbar = 2.2e-4 
    
    cell_size = box_size / n_grid  # Mpc/h per cell
    cell_volume = cell_size**3
    rms_cell = 1 / np.sqrt(nbar * cell_volume)
    
    for i in range(n_fields):
        
        if i % 100 == 0:
            print(f"Generating noise field {i}/{n_fields}")

        rng = np.random.default_rng(seed=seed_base+i)
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
    
    print(f"Generated {n_fields} noise fields in {dir_noise}")



def make_noise_fields_unit(n_fields, dir_noise, seed_base, overwrite=False):

    print(f"Generating n_fields={n_fields} unit noise fields in {dir_noise}")
    n_grid = 128
    failed_count = 0
    
    for i in range(n_fields):
        
        fn_noise = f"{dir_noise}/noise_field_n{i}.npy"
        if Path(fn_noise).exists() and not overwrite:
            continue
            
        if i % 100 == 0:
            print(f"Generating noise field {i}/{n_fields} (Failed: {failed_count})")
        
        rng = np.random.default_rng(seed=seed_base+i)
        noise_field = rng.standard_normal((n_grid, n_grid, n_grid))
        
        # having io issues; just running many times and it eventually fills them all in
        try:
            np.save(fn_noise, noise_field)
            print(f"Saved noise field {i}/{n_fields} to {fn_noise}")
        except (OSError, IOError) as e:
            print(f"I/O ERROR field {i}: {e} - SKIPPING")
            failed_count += 1
            # Remove partial file if it exists
            Path(fn_noise).unlink(missing_ok=True)
            continue

    print(f"Completed with {failed_count} failed saves")
        


if __name__ == "__main__":
    main()