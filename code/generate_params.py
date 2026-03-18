"""Generate Latin-hypercube and nested-LH parameter files for cosmo, bias, or noise."""
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import time

from scipy.stats import qmc
import gethypercube

import utils
import utils_cosmo


# Named parameter sets for reproducible Latin-hypercube runs.
PARAM_SETS_LH = {
    # Main cosmology LH
    "cosmo_p5_n10000": dict(
        bounds_type="cosmo",
        n_params_vary=5,
        n_samples=10_000,
        tag_bounds="",
        seed=42,
    ),
    # Test cosmology LH over "coverage" bounds
    "cosmo_coverage_p5_n10000": dict(
        bounds_type="cosmo",
        n_params_vary=5,
        n_samples=1_000,
        tag_bounds="_coverage",
        seed=42,
    ),
    # Test cosmology LH at fixed shame cosmology
    "cosmo_shame_p5_n1000": dict(
        bounds_type="cosmo",
        n_params_vary=5,
        n_samples=1_000,
        tag_bounds="_shame",
        seed=42,
    ),
    # Bias SHAMe, single fixed point
    "bias_shame_p0_n1": dict(
        bounds_type="bias",
        n_params_vary=0,
        n_samples=1,
        tag_bounds="_biasshame",
        seed=42,
    ),
    # Bias SHAMe, bl=0, single fixed point 
    "bias_shamebl0_p0_n1": dict(
        bounds_type="bias",
        n_params_vary=0,
        n_samples=1,
        tag_bounds="_biasshamebl0",
        seed=42,
    ),
    # Example: bias coverage run (uncomment/adjust values as needed)
    "bias_coverage_p4_n1000": dict(
        bounds_type="bias",
        n_params_vary=4,
        n_samples=1000,
        tag_bounds="_biascoverage",
        seed=53,
    ),
}

# Named parameter sets for nested LH bias runs.
PARAM_SETS_NESTED = {
    "biasnest_p4_n320000": dict(
        n_cosmo=10_000,
        n_factors=utils.n_factor_arr,  # [1,2,4,8,16,32]
        n_params_vary=4,
        tag_bounds="_biasnest",
        seed=42,
    ),
}


def main():
    # Select which LH parameter set to generate.
    scenario_name = "bias_shamebl0_p0_n1"
    config = PARAM_SETS_LH[scenario_name]
    generate_params_LH(**config)
    # Example for nested LH:
    # nested_name = "biasnest_p4_n320000"
    # nested_cfg = PARAM_SETS_NESTED[nested_name]
    # generate_params_nested_LH(**nested_cfg)
    # generate_params_fisher()


def generate_params_LH(
    bounds_type: str,
    n_params_vary: int,
    n_samples: int,
    tag_bounds: str,
    seed: int,
    overwrite: bool = False,
):
    """Generate LH (and fixed) param files for a given configuration."""
    if bounds_type == 'cosmo':
        param_names_vary = ['omega_cold', 'sigma8_cold', 'hubble', 'omega_baryon', 'ns']
        # for now we will always select the varying params in this order, for reproducibility
        param_names_vary = param_names_vary[:n_params_vary]
        param_names_ordered, bounds_dict, fiducial_dict = define_LH_cosmo(tag_bounds=tag_bounds)
    elif bounds_type == 'bias':
        param_names_vary = ['b1', 'b2', 'bs2', 'bl']
        param_names_vary = param_names_vary[:n_params_vary]
        param_names_ordered, bounds_dict, fiducial_dict = define_LH_bias(tag_bounds=tag_bounds)
    elif bounds_type == 'Anoise':
        param_names_vary = ['An_homog', 'An_b1', 'An_b2', 'An_bs2', 'An_bl']
        param_names_vary = param_names_vary[:n_params_vary]
        param_names_ordered, bounds_dict, fiducial_dict = define_LH_Anoise(tag_bounds=tag_bounds)
    else:
        raise ValueError("bounds_type must be 'cosmo' or 'bias' or 'Anoise'")
    
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
    

def generate_params_nested_LH(
    n_cosmo: int,
    n_factors,
    n_params_vary: int,
    tag_bounds: str,
    seed: int,
    overwrite: bool = False,
):
    """Generate nested LH bias params for a configurable design."""
    n_factors = np.array(n_factors)
    m_layers = list(n_cosmo * n_factors)

    # bias
    param_names_vary = ['b1', 'b2', 'bs2', 'bl']
    param_names_ordered, bounds_dict, fiducial_dict = define_LH_bias(tag_bounds=tag_bounds)
    
    n_total = m_layers[-1]
    tag_params = f'{tag_bounds}_p{n_params_vary}_n{n_total}'
    dir_params = '../data/params'
    fn_params = f'{dir_params}/params_lh{tag_params}.txt'
    fn_params_fixed = f'{dir_params}/params_fixed{tag_params}.txt'
    Path.mkdir(Path(dir_params), parents=True, exist_ok=True)

    if os.path.isfile(fn_params) and not overwrite:
        print(f'Found existing nested LH params: {fn_params}; stopping!')
        return
    
    if os.path.isfile(fn_params_fixed) and not overwrite:
        print(f'Found existing fixed params: {fn_params_fixed}; stopping!')
        return

    generate_nested_LH(param_names_vary, bounds_dict, m_layers, fn_params, seed=seed)
    fn_rands = f'{dir_params}/randints{tag_params}.npy'
    utils.generate_randints(n_total, fn_rands)
        
    param_names_fixed = [pn for pn in param_names_ordered if pn not in param_names_vary]
    if len(param_names_fixed) > 0:
        save_fixed_params(param_names_fixed, fn_params_fixed, fiducial_dict)
        


# TODO update with Anoise?? not sure if want to
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


def define_LH_cosmo(tag_bounds=''):
    
    param_names_ordered = ['omega_cold', 'sigma8_cold', 'hubble', 'omega_baryon', 'ns', 'neutrino_mass', 'w0', 'wa']

    # bounds from baccoemu for biased tracers (private version)
    # https://baccoemu.readthedocs.io/en/latest/#parameter-space
    bounds_dict = {'omega_cold'     :  [0.23, 0.4],
                    'omega_baryon'  :  [0.04, 0.06],
                    'sigma8_cold'   :  [0.65, 0.9], # public emulator only goes down to 0.73, but private down to 0.65
                    'ns'            :  [0.92, 1.01],
                    'hubble'        :  [0.6, 0.8],
                    'neutrino_mass' :  [0.0, 0.4],
                    'w0'            :  [-1.15, -0.85],
                    'wa'            :  [-0.3, 0.3],
                    }

    if tag_bounds == '_shame':
        fiducial_dict = utils.cosmo_dict_shame
    else:
        # this will be the fiducial for the LHs, if we fix some params
        fiducial_dict = utils.cosmo_dict_quijote

    return param_names_ordered, bounds_dict, fiducial_dict


def define_LH_bias(tag_bounds='_biasnest'):
    """Return (param_names_ordered, bounds_dict, fiducial_dict) for bias LH."""
    # biasnest
    bounds_dict = {'b1'     :  [-1.0, 3.0], #upped b1 max from 2 to 3
                    'b2'    :  [-2.0, 2.0],
                    'bs2'   :  [-2.0, 2.0],
                    'bl'   :  [-10.0, 10.0],
                }

    if tag_bounds=='_biasshame':
        fiducial_dict = utils.bias_dict_shame['_nbar0.00022']  # fiducial number density
    elif tag_bounds=='_biasshamebl0':
        fiducial_dict = utils.bias_dict_shame['_nbar0.00022_bl0']
    else:
        fiducial_dict = {'b1'     :  1.0,
                        'b2'    :  0.0,
                        'bs2'   :  0.0,
                        'bl'   :  0.0,
                        }
        
    return utils.biasparam_names_ordered, bounds_dict, fiducial_dict
    


def define_LH_Anoise(tag_bounds=''):
    """
    Define the parameter space for the noise field amplitude.
    """
    
    if tag_bounds is None:
        return [], {}, {}
    
    if tag_bounds == '_An':
        bounds_dict = {'A_noise': [0.0, 2.0]}
        fiducial_dict = {'A_noise': 1.0}
    elif '_Anmult' in tag_bounds:
        # same as b1
        bounds_dict = {'An_homog':  [-3.0, -3.0],
                       'An_b1'   :  [-3.0, 3.0],
                       'An_b2'   :  [-2.0, 2.0],
                       'An_bs2'  :  [-5.0, 5.0], 
                       'An_bl'   :  [-10.0, 10.0],
                      }
        fiducial_dict = {'An_homog':  1.0,
                         'An_b1'   :  0.0,
                         'An_b2'   :  0.0,
                         'An_bs2'  :  0.0,
                         'An_bl'   :  0.0,
                        }
    else:
        raise ValueError(f'Unknown tag_bounds {tag_bounds}')
    
    return utils.noiseparam_names_ordered, bounds_dict, fiducial_dict


def generate_LH(param_names_vary, bounds_dict, n_samples, fn_params, seed=42):
    """Sample Latin hypercube and save CSV to fn_params."""
    print(param_names_vary)

    n_params = len(param_names_vary)
    # using an old version of scipy for pytorch compatibility, and it takes seed not rng
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    # sample has shape (n_samples, n_params)
    sample = sampler.random(n=n_samples)
    # think it's fine to use same seed here
    rng = np.random.default_rng(seed)
    # default axis=0 which is shuffling in samples, as we want
    rng.shuffle(sample) # to ensure a random order (output is semi-random but not guaranteed)

    l_bounds = [bounds_dict[pn][0] for pn in param_names_vary]
    u_bounds = [bounds_dict[pn][1] for pn in param_names_vary]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

    param_df = pd.DataFrame()
    for param_name in param_names_vary:
        param_df[param_name] = sample_scaled[:,param_names_vary.index(param_name)]
    param_df.to_csv(fn_params)
    print(f'Saved LH to {fn_params}')


def generate_nested_LH(param_names_vary, bounds_dict, m_layers, fn_params, seed=42):
    """Generate a nested Latin hypercube design and save as a single file.
    
    Saves one CSV with columns for each bias param plus metadata columns
    ``idx_cosmo`` (which cosmology each row is assigned to) and ``nest_layer``
    (which nesting level the row was introduced in). Rows are in natural
    gethypercube order (first N rows = layer for N), with rows shuffled
    within each chunk of n_cosmo to break any ordering artifacts.
    """
    k = len(param_names_vary)
    n_cosmo = m_layers[0]
    n_total = m_layers[-1]
    max_factor = n_total // n_cosmo

    print(f'Generating nested LHD: k={k}, m_layers={m_layers}')
    t0 = time.time()
    layers = gethypercube.nested_lhd(k=k, m_layers=m_layers, seed=17,
                                     scramble=True, optimization=None)
    t1 = time.time()
    print(f'  nested_lhd: {t1 - t0:.2f}s')

    l_bounds = [bounds_dict[pn][0] for pn in param_names_vary]
    u_bounds = [bounds_dict[pn][1] for pn in param_names_vary]
    layers_phys = gethypercube.scale_LH(layers, l_bounds, u_bounds)
    full_design = layers_phys[-1]
    t2 = time.time()
    print(f'  scale_LH: {t2 - t1:.2f}s')

    # Shuffle rows within each chunk of n_cosmo to break ordering artifacts
    # from the LH construction, while preserving nesting (point sets unchanged)
    rng = np.random.default_rng(seed)
    for chunk in range(max_factor):
        start = chunk * n_cosmo
        end = start + n_cosmo
        perm = rng.permutation(n_cosmo)
        full_design[start:end] = full_design[start:end][perm]
    t3 = time.time()
    print(f'  per-chunk shuffle ({max_factor} chunks): {t3 - t2:.2f}s')

    # idx_cosmo: row i within each chunk maps to cosmo i
    idx_cosmo = np.tile(np.arange(n_cosmo), max_factor)

    # nest_layer: which nesting level each row was first introduced in
    n_factors = [m // n_cosmo for m in m_layers]
    nest_layer = np.empty(n_total, dtype=int)
    prev_factor = 0
    for factor in n_factors:
        nest_layer[prev_factor * n_cosmo : factor * n_cosmo] = utils.n_factor_to_nest_level[factor]
        prev_factor = factor

    param_df = pd.DataFrame(full_design, columns=param_names_vary)
    param_df['idx_cosmo'] = idx_cosmo
    param_df['nest_layer'] = nest_layer
    t4 = time.time()
    print(f'  build metadata + DataFrame: {t4 - t3:.2f}s')

    param_df.to_csv(fn_params)
    t5 = time.time()
    print(f'  save CSV: {t5 - t4:.2f}s')
    print(f'  TOTAL: {t5 - t0:.2f}s')
    print(f'Saved nested LH ({n_total} rows, {len(m_layers)} layers) to {fn_params}')


def save_fixed_params(param_names_fixed, fn_params_fixed, fiducial_dict):
    """Write single-row CSV of fixed parameters from fiducial_dict."""
    param_df_fixed = pd.DataFrame()    
    for param_name in param_names_fixed:
        param_df_fixed[param_name] = [fiducial_dict[param_name]]
    param_df_fixed.to_csv(fn_params_fixed, index=False)
    print(f'Saved fixed params to {fn_params_fixed}')




if __name__ == '__main__':
    main()