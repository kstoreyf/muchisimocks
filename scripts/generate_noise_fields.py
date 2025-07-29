from fileinput import filename
import numpy as np
import pandas as pd
from pathlib import Path
import re

import utils


def main():
    
    tag_noise = '_noise_quijote_p0_n1000'
    overwrite = False

    pattern = r'n(\d+)'
    match = re.search(pattern, tag_noise)

    if match:
        n_fields = int(match.group(1))
    else:
        raise ValueError(f"Pattern '{pattern}' not found in tag '{tag_noise}'")
    
    dir_noise = f'/scratch/kstoreyf/muchisimocks/data/noise_fields/fields{tag_noise}'
    Path(dir_noise).mkdir(parents=True, exist_ok=True)

    make_noise_fields(n_fields, dir_noise, overwrite=overwrite)


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
        noise_field = rms_cell * rng.standard_normal((n_grid, n_grid, n_grid))
        noise_field /= n_grid**3
        
        # Save the noise field to a file
        fn_noise = f"{dir_noise}/noise_field_n{i}.npy"
        if Path(fn_noise).exists() and not overwrite:
            print(f"File {fn_noise} already exists. Skipping...")
            continue
        else:
            np.save(fn_noise, noise_field)
            print(f"Saved noise field {i}/{n_fields} to {fn_noise}")


if __name__ == "__main__":
    main()