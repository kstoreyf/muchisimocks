import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.stats import qmc

import bacco

import data_loader
import utils


def main():

    n_data = 1000
    #tag_params = f'_p5_n{n_data}'
    # TODO change to tag_params, & tag_biasparams ?
    #tag_params = f'_test_p5_n{n_data}'
    
    #tag_params = f'_p2_n{n_data}'
    #tag_biasparams = '_b1000_p0_n1'
      
    tag_params = f'_quijote_p0_n{n_data}'
    tag_biasparams = '_b1000_p0_n1'  
    
    #tag_biasparams = '_biaszen_p4_n1000'
    tag_mocks = tag_params + tag_biasparams
    #tag_params = f'_fixedcosmo_n{n_data}'

    box_size = 1000.
    tag_errG = f'_boxsize{int(box_size)}'
    n_rlzs_per_cosmo = 1
    tag_datagen = f'{tag_errG}_nrlzs{n_rlzs_per_cosmo}'

    dir_emuPk = f'../data/emuPks/emuPks{tag_mocks}'
    Path.mkdir(Path(dir_emuPk), parents=True, exist_ok=True)
    
    # these emupks only depend on the params; can gen diff noisy versions from those
    # the function will check if we already have the emupks saved
    fn_emuPk = f'{dir_emuPk}/emuPks.npy'
    fn_emuPkerrG = f'{dir_emuPk}/emuPks_errgaussian{tag_errG}.npy'
    fn_emuPk_noisy = f'{dir_emuPk}/emuPks_noisy{tag_datagen}.npy'
    fn_emuk = f'{dir_emuPk}/emuPks_k.txt'
    # TODO separate randits for cosmo & bias, how to deal with?
    fn_rands = f'{dir_emuPk}/randints.npy'
    
    ### Read in parameters
    dir_params = '../data/params'
    fn_params = f'{dir_params}/params_lh{tag_params}.txt'
    fn_params_fixed = f'{dir_params}/params_fixed{tag_params}.txt'
    fn_biasparams = f'{dir_params}/params_lh{tag_biasparams}.txt'
    fn_biasparams_fixed = f'{dir_params}/params_fixed{tag_biasparams}.txt'
    
    params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, _, _ = data_loader.load_params(tag_params, tag_biasparams)
    # TODO could do multiple biases per cosmo, but for now 1:1
    assert params_df is None or len(params_df) == n_data, f"Expected {n_data} rows in {fn_params}"
    assert biasparams_df is None or len(biasparams_df) == n_data, f"Expected {n_data} rows in {fn_biasparams}"
        
    # copy parameter files to emu dict, to keep it all together / have duplicates
    fns_params = [fn_params, fn_params_fixed, fn_biasparams, fn_biasparams_fixed, fn_rands]
    for fn in fns_params:
        if not os.path.exists(fn):
            fn_nodir = Path(fn).name
            os.system(f'cp {fn} {dir_emuPk}/{fn_nodir}')

    ### Load emulator
    dir_emus_lbias = '/home/kstoreyf/external'
    emu, emu_bounds, emu_param_names = utils.load_emu(dir_emus_lbias=dir_emus_lbias)
    
    k, Pk_noiseless = generate_pks(emu, params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, 
                                   fn_emuk=fn_emuk, fn_emuPk=fn_emuPk, n_data=n_data,
                         )
    gaussian_error_pk = compute_noise(k, Pk_noiseless, box_size, fn_emuPkerrG=fn_emuPkerrG)
    draw_noisy_pk_realizations(Pk_noiseless, gaussian_error_pk, 
                               n_rlzs_per_cosmo=n_rlzs_per_cosmo,
                               fn_emuPk_noisy=fn_emuPk_noisy)

    
#def generate_pks(theta, bias_params, param_names, emu,
#                 fn_emuk=None, fn_emuPk=None, mode_bias_vector='single', overwrite=False):
def generate_pks(emu, params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed,
                 fn_emuk=None, fn_emuPk=None, n_data=None, overwrite=False):
    
    if os.path.exists(fn_emuk) and os.path.exists(fn_emuPk) and not overwrite:
        print(f"Loading from {fn_emuk} and {fn_emuPk} (already exist)")
        k = np.genfromtxt(fn_emuk)
        Pk = np.load(fn_emuPk)
        return k, Pk
    
    print(f"Generating emuPks for {fn_emuPk}...")
    #cosmo_params = utils.setup_cosmo_emu(cosmo='quijote')
    
    biasparam_names_ordered = ['b1', 'b2', 'bs2', 'bl']
    
    param_dict_fixed['expfactor'] = 1
    #cosmo_params['expfactor'] = 1
    #k = np.logspace(-2, np.log10(0.75), 30)
    # same as using for muchisimocks
    k = np.logspace(np.log10(0.01), np.log10(0.4), 30)

    Pk = []
    # if fixed cosmo and bias, we have to set n_data; do in main func
    if n_data is None:
        n_data = len(params_df)
    for idx_mock in range(n_data):
        param_dict = param_dict_fixed.copy()
        if params_df is not None:
            param_dict.update(params_df.loc[idx_mock].to_dict())
        
        biasparam_dict = biasparams_dict_fixed.copy()
        if biasparams_df is not None:
            biasparam_dict.update(biasparams_df.loc[idx_mock].to_dict())
        bias_vector = [biasparam_dict[name] for name in biasparam_names_ordered]

        _, pk_gg, _ = emu.get_galaxy_real_pk(bias=bias_vector, k=k, 
                                                    **param_dict)
        Pk.append(pk_gg)

    np.save(fn_emuPk, Pk)
    np.savetxt(fn_emuk, k)
    return np.array(k), np.array(Pk)


def compute_noise(k, Pk, box_size, fn_emuPkerrG=None, overwrite=False):
    
    if os.path.exists(fn_emuPkerrG) and not overwrite:
        print(f"Loading from {fn_emuPkerrG} (already exists)")
        gaussian_error_pk = np.load(fn_emuPkerrG, allow_pickle=True)
        return gaussian_error_pk
        
    print(f"Computing gaussian error for {fn_emuPkerrG}...")
    gaussian_error_pk = []
    for i in range(Pk.shape[0]):
        if i%100==0:
            print(i)
        g_err = bacco.statistics.approx_pk_gaussian_error(k, Pk[i], box_size)
        gaussian_error_pk.append(g_err)
    gaussian_error_pk = np.array(gaussian_error_pk)

    np.save(fn_emuPkerrG, gaussian_error_pk)
    return gaussian_error_pk


def draw_noisy_pk_realizations(Pk_noiseless, gaussian_error_pk, n_rlzs_per_cosmo=1,
                               rng=None, fn_emuPk_noisy=None, overwrite=False):
    
    if os.path.exists(fn_emuPk_noisy) and not overwrite:
        print(f"Loading from {fn_emuPk_noisy} (already exists)")
        Pk = np.load(fn_emuPk_noisy, allow_pickle=True)
        return Pk
    
    if rng is None:
        rng = np.random.default_rng(42)
        
    print(f"Drawing noisy Pk for {fn_emuPk_noisy}...")
    Pk = rng.normal(Pk_noiseless, gaussian_error_pk)    
    for _ in range(n_rlzs_per_cosmo-1):
        # i think first set should be equiv to orig?
        Pk_wnoise = rng.normal(Pk_noiseless, gaussian_error_pk)    
        Pk = np.vstack((Pk, Pk_wnoise))
    np.save(fn_emuPk_noisy, Pk)

    return Pk



if __name__=='__main__':
    main()
