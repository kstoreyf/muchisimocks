import numpy as np
import os
import pandas as pd
from pathlib import Path

from scipy.stats import qmc

import utils


def main():
    generate_params_LH()
    #generate_params_fisher()



def generate_params_LH():
    
    # seed 42, at least for the p5 mocks
    seed = 42
    overwrite = False # probs keep false!!! don't want to overwrite param files

    #n_samples = 1000000
    n_samples = 50000
    
    #bounds_type = 'cosmo' #cosmo or bias
    #n_params_vary = 0
    #tag_bounds = '' #NOTE params are automatically tagged below; this is for '_test' or '_quijote' so far
    #tag_bounds = '_test'
    #tag_bounds = '_quijote'
    
    # havent yet used 'test' to restrict bounds for bias params, only cosmo
    # TODO should i be?
    bounds_type = 'bias'
    n_params_vary = 4
    tag_bounds = '_biaszen'
    #n_params_vary = 1
    #tag_bounds = '_b1zen'
    #n_params_vary = 0
    #tag_bounds = '_b0000'
    
    if bounds_type == 'cosmo':
        param_names_vary = ['omega_cold', 'sigma8_cold', 'hubble', 'omega_baryon', 'ns']
        # for now we will always select the varying params in this order, for reproducibility
        param_names_vary = param_names_vary[:n_params_vary]
        param_names_ordered, bounds_dict, fiducial_dict = define_LH_cosmo(tag_bounds=tag_bounds)
    elif bounds_type == 'bias':
        param_names_vary = ['b1', 'b2', 'bs2', 'bl']
        param_names_vary = param_names_vary[:n_params_vary]
        param_names_ordered, bounds_dict, fiducial_dict = define_LH_bias(tag_bounds=tag_bounds)
    else:
        raise ValueError("pset must be 'cosmo' or 'bias'")
    
    tag_params = f'{tag_bounds}_p{len(param_names_vary)}_n{n_samples}'
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
    
    if len(param_names_vary) > 0:
        generate_LH(param_names_vary, bounds_dict, 
                    n_samples, fn_params, seed=seed)
        fn_rands = f'{dir_params}/randints{tag_params}.npy'
        utils.generate_randints(n_samples, fn_rands)
        
    param_names_fixed = [pn for pn in param_names_ordered if pn not in param_names_vary]
    if len(param_names_fixed) > 0:
        save_fixed_params(param_names_fixed, fn_params_fixed, fiducial_dict)
    

def generate_params_fisher():
    
    overwrite = False # probs keep false!!! don't want to overwrite param files
    delta_fracextent = 0.05
    n_deltas_per_side = 2
    
    #bounds_type = 'cosmo' #cosmo or bias
    bounds_type = 'bias'
    
    #n_params_vary = 0
    #tag_bounds = '' #NOTE params are automatically tagged below; this is for '_test' or '_quijote' so far
    #tag_bounds = '_test'
    #n_params_vary = 5
    #tag_bounds = '_quijote'
    
    # havent yet used 'test' to restrict bounds for bias params, only cosmo
    # TODO should i be?
    # bounds_type = 'bias'
    n_params_vary = 4
    tag_bounds = '_biaszen'
    #n_params_vary = 1
    #tag_bounds = '_b1zen'
    #n_params_vary = 0
    #tag_bounds = '_b1000'
    
    # for now choosing to not put pN in tag, bc usually will want to 
    # vary all of that set
    #tag_params = f'_fisher{tag_bounds}_p{len(param_names_vary)}
    tag_params = '_fisher'+tag_bounds
    
    dir_params = '../data/params'
    fn_params = f'{dir_params}/params{tag_params}.txt'
    if os.path.isfile(fn_params) and not overwrite:
        print(f'Found existing params file: {fn_params}; stopping!')
        return
    
    if bounds_type == 'cosmo':
        # for now we will always select the varying params in this order, for easy reproducibility
        param_names_vary = utils.cosmo_param_names_ordered[:n_params_vary]
        param_names_ordered, bounds_dict, fiducial_dict = define_LH_cosmo(tag_bounds=tag_bounds)
    elif bounds_type == 'bias':
        param_names_vary = utils.biasparam_names_ordered[:n_params_vary]
        param_names_ordered, bounds_dict, fiducial_dict = define_LH_bias(tag_bounds=tag_bounds)
    else:
        raise ValueError("pset must be 'cosmo' or 'bias'")
    
    ### Fisher deltas - changing one param at a time
    # Compute deltas: 5% of the extent of each parameter
    delta_units = []
    for pn in param_names_vary:
        extent = bounds_dict[pn][1] - bounds_dict[pn][0]
        delta_units.append(delta_fracextent * extent)
    delta_units = np.array(delta_units)
    
    # Prepare DataFrame
    rows = []
    
    # Fiducial cosmology
    # need float here bc in case the fiducial bias params are given as ints
    theta_fiducial = np.array([float(fiducial_dict[pn]) for pn in param_names_vary])
    # Add the fiducial to dataset
    row_fid = {p: v for p, v in zip(param_names_vary, theta_fiducial)}
    row_fid['param_shifted'] = 'fiducial'
    row_fid['n_deltas'] = 0
    rows.append(row_fid)
    
    # now add the deltas
    n_delta_arr = list(range(-n_deltas_per_side, 0)) + list(range(1, n_deltas_per_side+1))
    
    for i, pn in enumerate(param_names_vary):
        #for n_delta, delta in zip([0.5, 1.0], [0.5*deltas[i], deltas[i]]):
        for n_deltas in n_delta_arr:
            delta = n_deltas * delta_units[i]
            theta = theta_fiducial.copy()
            theta[i] += delta
            row = {p: v for p, v in zip(param_names_vary, theta)}
            row['param_shifted'] = pn
            row['n_deltas'] = n_deltas
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    #### Save
    Path(dir_params).mkdir(parents=True, exist_ok=True)
    df.to_csv(fn_params)
    print(f'Saved Fisher parameter grid to {fn_params}')    

    ### now generate and save fixed params
    fn_params_fixed = f'{dir_params}/params_fixed{tag_params}.txt'
    if os.path.isfile(fn_params_fixed) and not overwrite:
        print(f'Found existing fixed params: {fn_params_fixed}; stopping!')
        return
        
    param_names_fixed = [pn for pn in param_names_ordered if pn not in param_names_vary]
    if len(param_names_fixed) > 0:
        save_fixed_params(param_names_fixed, fn_params_fixed, fiducial_dict)




def define_LH_bias(tag_bounds='biaszen'):
        
    if 'biaswide' in tag_bounds:
        bounds_dict = {'b1'     :  [-5.0, 20.0],
                        'b2'    :  [-5.0, 10.0],
                        'bs2'   :  [-10.0, 20.0],
                        'bl'   :  [-20.0, 30.0],
                    }
    elif 'biaszen' in tag_bounds:
        bounds_dict = {'b1'     :  [-1.0, 2.0],
                        'b2'    :  [-2.0, 2.0],
                        'bs2'   :  [-2.0, 2.0],
                        'bl'   :  [-10.0, 10.0],
                    }
    elif 'b1zen' in tag_bounds:
        bounds_dict = {'b1'     :  [-1.0, 2.0],
                    }
    else:
        bounds_dict = {}
    
    if 'b0000' in tag_bounds:
        fiducial_dict = {'b1'     :  0.0,
                        'b2'    :  0.0,
                        'bs2'   :  0.0,
                        'bl'   :  0.0,
                    }
    else:
        # this will be fiducial for b1000, and all others
        fiducial_dict = {'b1'     :  1.0,
                        'b2'    :  0.0,
                        'bs2'   :  0.0,
                        'bl'   :  0.0,
                    }
    
    # used to make the separate test sets, to avoid edge effects
    if 'test' in tag_bounds:
        bounds_dict = restrict_bounds(bounds_dict, factor=0.05)
        
    return utils.biasparam_names_ordered, bounds_dict, fiducial_dict
    
    
def define_LH_cosmo(tag_bounds=''):
    
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
    
    if 'test' in tag_bounds:
        bounds_dict = restrict_bounds(bounds_dict, factor=0.05)
    
    fiducial_dict = utils.cosmo_dict_quijote
    # not checking for bounds bc now these are the only two options (these bounds and quijote fid)
    # if tag_bounds == '_quijote':
    #     fiducial_dict = utils.cosmo_dict_quijote
    # else:
    #     raise ValueError(f'Unknown tag_bounds {tag_bounds}')

    return param_names_ordered, bounds_dict, fiducial_dict


def restrict_bounds(bounds_dict, factor=0.05):
    # for test set, reduce edges by 5%
    bounds_dict_reduced = {}
    for name in bounds_dict.keys():
        l_bound, u_bound = bounds_dict[name]
        width = u_bound - l_bound
        l_bound = l_bound + factor * width
        u_bound = u_bound - factor * width
        bounds_dict_reduced[name] = [l_bound, u_bound]
    return bounds_dict_reduced


def generate_LH(param_names_vary, bounds_dict, 
                n_samples, fn_params, 
                seed=42):

    print(param_names_vary)

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


def save_fixed_params(param_names_fixed, fn_params_fixed, fiducial_dict):
    param_df_fixed = pd.DataFrame()    
    for param_name in param_names_fixed:
        param_df_fixed[param_name] = [fiducial_dict[param_name]]
    param_df_fixed.to_csv(fn_params_fixed, index=False)
    print(f'Saved fixed params to {fn_params_fixed}')




if __name__ == '__main__':
    main()