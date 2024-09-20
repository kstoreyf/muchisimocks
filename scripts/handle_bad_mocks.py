import glob
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import pandas as pd
#import pyfftw
import re

import bacco
import baccoemu

import sys
sys.path.append('/dipc/kstoreyf/muchisimocks/scripts')
import utils


def main():
    identify_bad_mocks()
    #remove_bad_mocks()
    
def remove_bad_mocks():
    dry_run = False
    tag_params = '_p5_n10000'
    tag_mocks = tag_params
    dir_mocks = f'/scratch/kstoreyf/muchisimocks/muchisimocks_lib{tag_mocks}'
    fn_idxs_bad = f'../data/idxs_LH_bad{tag_params}.txt'    
    idxs_LH_bad = np.loadtxt(fn_idxs_bad, dtype=int)
    print(idxs_LH_bad)
    
    for idx_LH in idxs_LH_bad:
        dir_LH = f'{dir_mocks}/LH{idx_LH}'
        print(f"Removing files from {dir_LH}:")
        if not dry_run:
            # just remove files, to be safe
            fn_bfields_kcut_deconvolved = f'{dir_LH}/bias_fields_eul_deconvolved_{idx_LH}.npy'
            tag_zspace = '_zspace'
            fn_bfields_zspace_kcut_deconvolved = f'{dir_LH}/bias_fields_eul{tag_zspace}_deconvolved_{idx_LH}.npy'
            fn_disp = f'{dir_LH}/pred_disp.npy'
            fn_vel = f'{dir_LH}/pred_vel.npy'
            fn_ZA_disp = f'{dir_LH}/ZA_disp.npy'
            fn_lin = f'{dir_LH}/lin_field.npy'
            fn_ZA_vel = f'{dir_LH}/ZA_vel.npy'
            fn_params_m2m = f'{dir_LH}/cosmo_pars_m2m.txt'
            fns_to_remove = [fn_bfields_kcut_deconvolved, fn_bfields_zspace_kcut_deconvolved,
                   fn_disp, fn_vel, fn_ZA_disp, fn_lin, fn_ZA_vel, fn_params_m2m]
            for fn in fns_to_remove:
                if os.path.exists(fn):
                    os.remove(fn)
            print("Removed files (if they existed)!")
            # finally remove the directory - this will only work if empty
            if os.path.exists(dir_LH):
                os.rmdir(dir_LH)
                print("Removed dir")
        #break
            
            
def identify_bad_mocks():

    # load lib
    n_grid = 128
    box_size = 1000.0

    #tag_params = '_p3_n500'
    tag_params = '_p5_n10000'

    fn_idxs_bad = f'../data/idxs_LH_bad{tag_params}.txt'
    with open(fn_idxs_bad, 'w') as file:
        # just create file
        pass

    tag_mocks = tag_params
    dir_mocks = f'/scratch/kstoreyf/muchisimocks/muchisimocks_lib{tag_mocks}'

    fn_params = f'{dir_mocks}/params_lh{tag_params}.txt'
    fn_params_fixed = f'{dir_mocks}/params_fixed{tag_params}.txt'
    params_df = pd.read_csv(fn_params, index_col=0)
    param_dict_fixed = pd.read_csv(fn_params_fixed).loc[0].to_dict()

    #idxs_LH = list(params_df.index.values)
    # this one gets only ones we've computed pk of (zspace, cuz probs have both if have zspace)
    #idxs_LH_all = [int(re.search(r'pk_(\d+)\.npy', f).group(1)) for f in glob.glob(f'{dir_pks_zspace}/pk_*.npy')]

    # via chatgpt; * should get 
    files = glob.glob(f'{dir_mocks}/LH*/bias_fields_eul*_deconvolved_*.npy')
    idxs_LH_all = sorted({int(re.search(r'_(\d+)\.npy', f).group(1)) for f in files if re.search(r'_(\d+)\.npy', f)})

    idxs_LH_all = np.sort(np.array(idxs_LH_all))
    print(idxs_LH_all)

    #idxs_LH_check = idxs_LH_all
    idxs_LH_check = [2]

    n_threads_bacco = 16
    bacco.configuration.update({'pk':{'boltzmann_solver': 'CLASS'}})
    bacco.configuration.update({'pknbody' : {'ngrid'  :  n_grid}})
    bacco.configuration.update({'scaling' : {'disp_ngrid' : n_grid}})
    bacco.configuration.update({'number_of_threads': n_threads_bacco})
        
    idxs_LH_bad = []
    for idx_LH in idxs_LH_check:
        
        print(f"Checking {idx_LH}...", flush=True)
        
        # check if worth checking (if either field exists)
        # i think this is redundant w how im finding the idxs actually
        # if (not os.path.exists(fn_bfields_kcut_deconvolved)) and (not os.path.exists(fn_bfields_zspace_kcut_deconvolved)):
        #     print("Neither field exists, moving on")
        #     continue
        
        param_dict = params_df.loc[idx_LH].to_dict()
        param_dict.update(param_dict_fixed)
        expfactor = 1.0
        cosmo = utils.get_cosmo(param_dict, a_scale=expfactor, sim_name='quijote')

        ## Start cosmology class
        seed = idx_LH
        sim, disp_field = bacco.utils.create_lpt_simulation(cosmo, box_size, Nmesh=n_grid, Seed=seed,
                                                            FixedInitialAmplitude=False, InitialPhase=0, 
                                                            expfactor=expfactor, LPT_order=1, order_by_order=None,
                                                            phase_type=1, ngenic_phases=True, return_disp=True, 
                                                            sphere_mode=0)

        grid = bacco.visualization.uniform_grid(npix=n_grid, L=box_size, ndim=3, bounds=False)
        pos_ZA = bacco.scaler.add_displacement(None,
                                            disp_field,
                                            box=box_size,
                                            #pos=grid.reshape(-1,3),
                                            pos=grid.reshape(-1,3),
                                            vel=None,
                                            vel_factor=0,
                                            verbose=True)[0]
        pos_ZA_mesh = bacco.statistics.compute_mesh(ngrid=n_grid, box=box_size, pos=pos_ZA, 
                                            deposit_method='cic', interlacing=False)[0]
        pos_ZA_mesh_normed = (pos_ZA_mesh - np.mean(pos_ZA_mesh))/np.std(pos_ZA_mesh)

        # load map2map fields
        dir_LH = f'{dir_mocks}/LH{idx_LH}'
        fn_bfields_kcut_deconvolved = f'{dir_LH}/bias_fields_eul_deconvolved_{idx_LH}.npy'
        if os.path.exists(fn_bfields_kcut_deconvolved):
            
            bias_terms_eul_pred_kcut_deconvolved = np.load(fn_bfields_kcut_deconvolved)
            
            # write metric for comparison
            bias0 = bias_terms_eul_pred_kcut_deconvolved[0]
            bias0_normed = (bias0 - np.mean(bias0))/np.std(bias0)
            dist_mean = np.mean((bias0_normed-pos_ZA_mesh_normed)**2)
            # found 1 empirically as good cutoff
            if dist_mean > 1:
                print(f"BAD {idx_LH}", flush=True)
                idxs_LH_bad.append(idx_LH)
                
                with open(fn_idxs_bad, 'a') as file:
                    file.write(f"{idx_LH}\n")
                    
                # don't need to check zspace if real space is bad, gonna redo whole thing
                continue
            else:
                print(f"GOOD! {idx_LH}", flush=True)
            
        tag_zspace = '_zspace'
        fn_bfields_zspace_kcut_deconvolved = f'{dir_LH}/bias_fields_eul{tag_zspace}_deconvolved_{idx_LH}.npy'
        if os.path.exists(fn_bfields_zspace_kcut_deconvolved):
            
            bias_terms_eul_pred_zspace_kcut_deconvolved = np.load(fn_bfields_zspace_kcut_deconvolved)

            bias0_zspace = bias_terms_eul_pred_zspace_kcut_deconvolved[0]
            bias0_zspace_normed = (bias0_zspace - np.mean(bias0_zspace))/np.std(bias0_zspace)
            
            dist_zspace_mean = np.mean((bias0_zspace_normed-pos_ZA_mesh_normed)**2)

            if dist_zspace_mean > 1:
                print(f"BAD {idx_LH} (zspace)", flush=True)
                idxs_LH_bad.append(idx_LH)
                
                with open(fn_idxs_bad, 'a') as file:
                    file.write(f"{idx_LH}\n")
            else:
                print(f"GOOD! {idx_LH} (zspace)", flush=True)
        
        
if __name__ == '__main__':
    main()