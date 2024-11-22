import numpy as np
import os
import pandas as pd
from pathlib import Path

from scipy.stats import qmc

import utils



def main():
    
    # seed 42, at least for the p5 mocks
    seed = 42
    overwrite = False

    n_samples = 10000
    
    pset = 'biaszen'
    #pset = 'cosmo'
    
    if pset=='cosmo':
        #param_names_vary = ['omega_cold', 'sigma8_cold', 'hubble']
        param_names_vary = ['omega_cold', 'sigma8_cold', 'hubble', 'omega_baryon', 'ns']
        param_names_ordered, bounds_dict, fiducial_dict = define_LH_cosmo()
    elif 'bias' in pset:
        #param_names_vary = ['b1', 'b2', 'bs2', 'bl']
        param_names_vary = ['b1']
        param_names_ordered, bounds_dict, fiducial_dict = define_LH_bias(bounds=pset)
    else:
        raise ValueError("pset must be 'cosmo' or 'bias'")
    
    tag_params = f'_{pset}_p{len(param_names_vary)}_n{n_samples}'
    dir_params = '../data/params'
    fn_params = f'{dir_params}/params_lh{tag_params}.txt'
    fn_params_fixed = f'{dir_params}/params_fixed{tag_params}.txt'
    Path.mkdir(Path(dir_params), parents=True, exist_ok=True)
    
    if os.path.isfile(fn_params) and not overwrite:
        print(f'Found existing LH params: {fn_params}; stopping!')
        return
    
    if os.path.isfile(fn_params_fixed) and not overwrite:
        print(f'Found existing fixed params: {fn_params_fixed}; stopping!')
        return
    
    generate_LH(param_names_vary, param_names_ordered, bounds_dict, 
                n_samples, fn_params, fn_params_fixed, 
                fiducial_dict=fiducial_dict, seed=seed)
    
    fn_rands = f'{dir_params}/randints{tag_params}.npy'
    utils.generate_randints(n_samples, fn_rands)


def define_LH_bias(bounds='mid'):
    
    param_names_ordered = ['b1', 'b2', 'bs2', 'bl']
    
    if bounds=='bias':
        bounds_dict = {'b1'     :  [-5, 20],
                        'b2'    :  [-5, 10],
                        'bs2'   :  [-10, 20],
                        'bl'   :  [-20, 30],
                    }
    elif bounds=='biaszen':
        bounds_dict = {'b1'     :  [-1, 2],
                        'b2'    :  [-2, 2],
                        'bs2'   :  [-2, 2],
                        'bl'   :  [-10, 10],
                    }
    else:
        raise ValueError("bounds must be 'wide' or 'mid'")
    
    fiducial_dict = {'b1'     :  1,
                    'b2'    :  0,
                    'bs2'   :  0,
                    'bl'   :  0,
                }
    
    return param_names_ordered, bounds_dict, fiducial_dict
    
    
def define_LH_cosmo():
    
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

    return param_names_ordered, bounds_dict, fiducial_dict



def generate_LH(param_names_vary, param_names_ordered, bounds_dict, 
                n_samples, fn_params, fn_params_fixed, 
                fiducial_dict=None,
                seed=42):

    param_names_fixed = [pn for pn in param_names_ordered if pn not in param_names_vary]
    print(param_names_vary)
    print(param_names_fixed)

    n_params = len(param_names_vary)
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    sample = sampler.random(n=n_samples)

    l_bounds = [bounds_dict[pn][0] for pn in param_names_vary]
    u_bounds = [bounds_dict[pn][1] for pn in param_names_vary]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

    param_df = pd.DataFrame()
    for param_name in param_names_vary:
        param_df[param_name] = sample_scaled[:,param_names_vary.index(param_name)]
    param_df.to_csv(fn_params)
    print(f'Saved LH to {fn_params}')

    if fiducial_dict is not None:
        param_df_fixed = pd.DataFrame()    
        for param_name in param_names_fixed:
            param_df_fixed[param_name] = [fiducial_dict[param_name]]
        param_df_fixed.to_csv(fn_params_fixed, index=False)
        print(f'Saved LH to {param_df_fixed}')




if __name__ == '__main__':
    main()