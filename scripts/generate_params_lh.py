import numpy as np
import os
import pandas as pd
from pathlib import Path

from scipy.stats import qmc

import utils


seed = 42

n_samples = 10000
#param_names_vary = ['omega_cold', 'sigma8_cold', 'hubble']
param_names_vary = ['omega_cold', 'sigma8_cold', 'hubble', 'omega_baryon', 'ns']
tag_params = f'_p{len(param_names_vary)}_n{n_samples}'
dir_params = '../data/params'
fn_params = f'{dir_params}/params_lh{tag_params}.txt'
fn_params_fixed = f'{dir_params}/params_fixed{tag_params}.txt'

Path.mkdir(Path(dir_params), parents=True, exist_ok=True)

param_names_ordered = ['omega_cold', 'sigma8_cold', 'hubble', 'omega_baryon', 'ns', 'neutrino_mass', 'w0', 'wa']

# bounds from baccoemu for biased tracers (private version)
# https://baccoemu.readthedocs.io/en/latest/#parameter-space
bounds_dict = {'omega_cold'     :  [0.23, 0.4],
                'omega_baryon'  :  [0.04, 0.06],
                'sigma8_cold'   :  [0.65, 0.9], # public only goes down to 0.73, but w private down to 0.65
                'ns'            :  [0.92, 1.01],
                'hubble'        :  [0.6, 0.8],
                'neutrino_mass' :  [0.0, 0.4],
                'w0'            :  [-1.15, -0.85],
                'wa'            :  [-0.3, 0.3],
                }

fiducial_dict = utils.cosmo_dict_quijote

param_names_fixed = [pn for pn in param_names_ordered if pn not in param_names_vary]

n_params = len(param_names_vary)
sampler = qmc.LatinHypercube(d=n_params, seed=seed)
sample = sampler.random(n=n_samples)

l_bounds = [bounds_dict[pn][0] for pn in param_names_vary]
u_bounds = [bounds_dict[pn][1] for pn in param_names_vary]
sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

# param_df = pd.DataFrame()
# for param_name in param_names_ordered:
#     if param_name in param_names_vary:
#         vals = sample_scaled[:,param_names_vary.index(param_name)]
#     else:
#         vals = [fiducial_dict[param_name]]*n_samples
#     param_df[param_name] = vals
# param_df.to_csv(fn_params)

param_df = pd.DataFrame()
for param_name in param_names_vary:
    param_df[param_name] = sample_scaled[:,param_names_vary.index(param_name)]
param_df.to_csv(fn_params)

param_df_fixed = pd.DataFrame()    
for param_name in param_names_fixed:
    param_df_fixed[param_name] = [fiducial_dict[param_name]]
param_df_fixed.to_csv(fn_params_fixed, index=False)
