import os
os.environ["OMP_NUM_THREADS"] = str(1)

import numpy as np
import pandas as pd
from pathlib import Path
import re
import time

import bacco

import utils



def main():
    compute_pks_muchisimocks()
    #compute_pks_quijote_LH()
    

def compute_pks_muchisimocks():
    
    #tag_mocks = '_HR'
    #tag_mocks = '_FixedPk'
    
    #tag_params = '_p3_n10'
    #dir_mocks = f'../data/cosmolib{tag_mocks}'
    #dir_mocks = f'/cosmos_storage/cosmosims/muchisimocks_lib{tag_mocks}'
    
    # p5_n50 is in data, bc contains full fields as checks so needs more space
    #tag_params = '_p5_n50'
    #tag_mocks = tag_params
    #dir_mocks = f'/data/kstoreyf/muchisimocks/muchisimocks_lib{tag_mocks}'

    n_threads = 16
    
    tag_params = '_p5_n10000'
    tag_mocks = tag_params
    dir_mocks = f'/scratch/kstoreyf/muchisimocks/muchisimocks_lib{tag_mocks}'
    
    # tag_pk = '_b0000'
    # tag_fields = '_deconvolved'
    # tag_pk = '_b0000_zspace'
    # tag_fields = '_zspace_deconvolved'
    
    bias_vector = [1., 0., 0., 0.]    
    tags_pk = ['_b1000', '_b1000_zspace']
    tags_fields = ['_deconvolved', '_zspace_deconvolved']
        
    idxs_LH = [0]
    # idxs_LH = np.sort([int(re.search(r'^LH(\d+)$', dir_mocks).group(1)) \
    #     for dir_mocks in os.listdir(dir_mocks) \
    #     if re.search(r'^LH\d+$', dir_mocks)])
    #idxs_LH = [43]
    
    #tag_fields = '_hr'
    #tag_fields_extra = '_2GpcBox'
    tag_fields_extra = ''
    overwrite = True
    
    deconvolve_grid = False # fields already deconvolved
    # NOTE: fields created without interlacing, so need to set it off here
    interlacing = False
    correct_grid = False
    k_min, k_max, n_bins = 0.01, 0.4, 30


    #n_grid = 128
    #n_grid_orig = None #compute from fields, if we don't know that it's different
    n_grid_orig = 512
    if '2Gpc' in tag_fields_extra:
        box_size = 2000.0
    else:
        box_size = 1000.0
    
    fn_params = f'{dir_mocks}/params_lh{tag_params}.txt'
    fn_params_fixed = f'{dir_mocks}/params_fixed{tag_params}.txt'
    params_df = pd.read_csv(fn_params, index_col=0)
    param_dict_fixed = pd.read_csv(fn_params_fixed).loc[0].to_dict()
    # order of saved cosmo param files
    #param_names = ['omega_cold', 'sigma_8', 'h', 'omega_baryon', 'n_s', 'seed']
    
    for tag_pk, tag_fields in zip(tags_pk, tags_fields):
        dir_pks = f'../data/pks_mlib/pks{tag_mocks}{tag_pk}'
        Path.mkdir(Path(dir_pks), parents=True, exist_ok=True)
        fn_bias_vector = f'{dir_pks}/bias_params.txt'
        np.savetxt(fn_bias_vector, bias_vector)
    
        print("tag_pk:", tag_pk)
        for idx_LH in idxs_LH:
            #if idx_LH%10==0:
            print(f"Comping Pk for LH{idx_LH} ({tag_pk})")
            fn_fields = f'{dir_mocks}/LH{idx_LH}/bias_fields_eul{tag_fields}_{idx_LH}{tag_fields_extra}.npy'
            #fn_params = f'{dir_mocks}/LH{idx_LH}/cosmo_{idx_LH}.txt'
            fn_pk = f'{dir_pks}/pk_{idx_LH}{tag_fields_extra}.npy'
            if os.path.exists(fn_pk) and not overwrite:
                print(f"P(k) for idx_LH={idx_LH} exists and overwrite={overwrite}, continuing")
                continue
            
            start = time.time()
            # normalize by 512 because that's the original ngrid size
            try:
                bias_terms_eul = np.load(fn_fields)
            except FileNotFoundError:
                print(f"File {fn_fields} not found, continuing")
                continue
            if n_grid_orig is None:
                n_grid_orig = bias_terms_eul.shape[-1]
            print(f"n_grid_orig = {n_grid_orig}")
            tracer_field = get_tracer_field(bias_terms_eul, bias_vector, n_grid_norm=n_grid_orig)
            
            #param_vals = np.loadtxt(fn_params)
            #param_dict = dict(zip(param_names, param_vals))
            param_dict = params_df.loc[idx_LH].to_dict()
            param_dict.update(param_dict_fixed)
            print(param_dict)
            cosmo = utils.get_cosmo(param_dict)
            
            compute_pk(tracer_field, cosmo, box_size,
                        k_min=k_min, k_max=k_max, n_bins=n_bins,
                        deconvolve_grid=deconvolve_grid,
                        interlacing=interlacing, correct_grid=correct_grid,
                        fn_pk=fn_pk,
                        n_threads=n_threads)
            end = time.time()
            print(f"Computed P(k) for idx_LH={idx_LH} ({tag_pk}) in time {end-start} s")
    

def compute_pks_quijote_LH():
    
    n_threads = 8
    
    dir_mocks = '/cosmos_storage/home/mpelle/Yin_data/Quijote'
    tag_fields = '_interlacingfalse_fixdamp'
    dir_fields = f'/cosmos_storage/home/kstoreyf/data_muchisimocks/quijote_LH{tag_fields}'
    tag_pk = f'{tag_fields}_b0000'

    #dir_pks_pred = f'../data/pks_quijote_LH/pks_pred{tag_pk}'
    overwrite_fields = False
    overwrite_pks = False
    #compute_sim = True
    #compute_pred = False
    tags = ['_sim', '_pred']
    #tags = ['_pred']
    
    #Path.mkdir(Path(dir_pks_pred), parents=True, exist_ok=True)

    bias_vector = [1., 0., 0., 0.]
    n_grid = 512
    n_grid_orig = 512
    box_size = 1000.0
    k_min, k_max, n_bins = 0.01, 1, 50

    k_nyq = np.pi * n_grid / box_size
    damping_scale = k_nyq
            
    deconvolve_grid = True
    interlacing = False #True
    correct_grid = False
    
    run_zspace = True
    
    # order of saved cosmo param files (via https://quijote-simulations.readthedocs.io/en/latest/LH.html)
    # careful that here it's omega_m, whereas for muchisimocks/cosmolib it's omega_cold
    # (should be handled in utils.get_cosmo() function)
    param_names = ['omega_m', 'omega_baryon', 'h', 'n_s', 'sigma_8']

    # idxs_LH = np.array([10,29,37,40,70,85,127,158,165,184,208,220,240,254,267,274,293,305,336,374,375,388,433,444,
    #                   464,502,534,542,574,598,605,628,652,663,676,700,702,721,737,762,809,822,825,837,853,864,882,
    #                   899,901,911,939,948,950,951,964,976,977,1016,1022,1041,1050,1060,1082,1091,1103,1114,1147,
    #                   1157,1173,1175,1219,1222,1299,1309,1314,1317,1331,1365,1372,1378,1391,1397,1418,1444,1459,
    #                   1510,1512,1513,1515,1517,1533,1553,1567,1568,1599,1622,1642,1657,1659,1667])
    #idxs_LH = idxs_LH[:1]
    idxs_LH = [663]

    #if compute_sim:
    for tag in tags:
    
        dir_pks = f'../data/pks_quijote_LH/pks{tag}{tag_pk}'
        dir_pks_zspace = f'../data/pks_quijote_LH/pks_zspace{tag}{tag_pk}'
        
        Path.mkdir(Path(dir_pks), parents=True, exist_ok=True)
        Path.mkdir(Path(dir_pks_zspace), parents=True, exist_ok=True)
        
        fn_bias_vector = f'{dir_pks}/bias_params.txt'
        np.savetxt(fn_bias_vector, bias_vector)
        fn_bias_vector = f'{dir_pks_zspace}/bias_params.txt'
        np.savetxt(fn_bias_vector, bias_vector)
        
        for idx_LH in idxs_LH:

            if idx_LH%10==0:
                print(idx_LH)
            idx_LH_str = f'{idx_LH:04}'
         
            # get params
            fn_params = f'{dir_mocks}/LH{idx_LH_str}/param_{idx_LH_str}.txt'
            param_vals = np.loadtxt(fn_params)
            param_dict = dict(zip(param_names, param_vals))
            cosmo = utils.get_cosmo(param_dict)
                  
            # check fields existence
            fn_fields = f'{dir_fields}/LH{idx_LH_str}/Eulerian_fields{tag}_{idx_LH_str}.npy'
            fn_fields_zspace = f'{dir_fields}/LH{idx_LH_str}/Eulerian_fields_zspace{tag}_{idx_LH_str}.npy'
            
            Path.mkdir(Path(f'{dir_fields}/LH{idx_LH_str}'), parents=True, exist_ok=True)

            fn_dens_lin = f'{dir_mocks}/LH{idx_LH_str}/lin_den_{idx_LH_str}.npy'
            dens_lin = np.load(fn_dens_lin)
            
            # get fields sim
            if 'sim' in tag:
                fn_disp = f'{dir_mocks}/LH{idx_LH_str}/dis_{idx_LH_str}.npy'
            elif 'pred' in tag:
                fn_disp = f'{dir_mocks}/LH{idx_LH_str}/pred_pos_{idx_LH_str}.npy'
            else:
                raise ValueError("tag must be 'sim' or 'pred'")
            disp = np.load(fn_disp)
            
            if (not os.path.exists(fn_fields) or overwrite_fields):
                start = time.time()
                print(f"Computing fields for orig sim for idx_LH={idx_LH}")
                bias_terms_eul = displacements_to_bias_fields(dens_lin, disp, n_grid, 
                                            box_size, damping_scale=damping_scale, interlacing=interlacing, fn_fields=fn_fields,
                                            n_threads=n_threads)
                end = time.time()
                print(f"Generated bias fields for orig sim for idx_LH={idx_LH} in time {end-start} s")
            else:
                print(f"Fields for orig sim for idx_LH={idx_LH} exists and overwrite={overwrite_fields}, loading")
                bias_terms_eul = np.load(fn_fields, allow_pickle=True)
                
            if run_zspace and (not os.path.exists(fn_fields_zspace) or overwrite_fields):
                start = time.time()
                if 'sim' in tag:
                    fn_vel = f'{dir_mocks}/LH{idx_LH_str}/nlvel_{idx_LH_str}.npy'
                elif 'pred' in tag:
                    fn_vel = f'{dir_mocks}/LH{idx_LH_str}/pred_vel_{idx_LH_str}.npy'
                else:
                    raise ValueError("tag must be 'sim' or 'pred'")
                vel = np.load(fn_vel)
                velocities = fv2bro(vel.copy(order='C'))
                bias_terms_eul_zspace = displacements_to_bias_fields(dens_lin, disp, n_grid, 
                                            box_size, velocities=velocities, cosmo=cosmo, damping_scale=damping_scale, 
                                            interlacing=interlacing, fn_fields=fn_fields_zspace,
                                            n_threads=n_threads)
                end = time.time()
                print(f"Generated zspace bias fields for orig sim for idx_LH={idx_LH} in time {end-start} s")
            else:
                print(f"Zspace fields for orig sim for idx_LH={idx_LH} exists and overwrite={overwrite_fields}, loading")
                bias_terms_eul_zspace = np.load(fn_fields_zspace, allow_pickle=True)
                
            # check pk existence
            fn_pk = f'{dir_pks}/pk_{idx_LH_str}.npy'
            if os.path.exists(fn_pk) and not overwrite_pks:
                print(f"P(k) for orig sim for idx_LH={idx_LH} exists and overwrite={overwrite_pks}, continuing")
            else:
                start = time.time()
                tracer_field = get_tracer_field(bias_terms_eul, bias_vector, n_grid_norm=n_grid_orig)
                compute_pk(tracer_field, cosmo, box_size,
                        k_min=k_min, k_max=k_max, n_bins=n_bins,
                        deconvolve_grid=deconvolve_grid,
                        interlacing=interlacing, correct_grid=correct_grid,
                        fn_pk=fn_pk)
                end = time.time()
                print(f"Computed P(k) for orig sim for idx_LH={idx_LH} in time {end-start} s")

            fn_pk_zspace = f'{dir_pks_zspace}/pk_{idx_LH_str}.npy'
            if os.path.exists(fn_pk_zspace) and not overwrite_pks:
                print(f"P(k) zspace for orig sim for idx_LH={idx_LH} exists and overwrite={overwrite_pks}, continuing")
            else:
                start = time.time()
                tracer_field_zspace = get_tracer_field(bias_terms_eul_zspace, bias_vector, n_grid_norm=n_grid_orig)
                compute_pk(tracer_field_zspace, cosmo, box_size,
                        k_min=k_min, k_max=k_max, n_bins=n_bins,
                        deconvolve_grid=deconvolve_grid,
                        interlacing=interlacing, correct_grid=correct_grid,
                        fn_pk=fn_pk_zspace)
                end = time.time()
                print(f"Computed P(k) zspace for orig sim for idx_LH={idx_LH} in time {end-start} s")


    
def remove_kmodes_and_downsample_quijote():
    pass
    
    
def displacements_to_bias_fields(dens_lin, disp, n_grid, box_size, 
                                 velocities=None, cosmo=None,
                                 damping_scale=None, n_threads=8,
                                 interlacing=True,
                                 fn_fields=None):
    
    if damping_scale is None:
        k_nyq = np.pi * n_grid / box_size
        damping_scale = k_nyq

    bacco.configuration.update({'number_of_threads': n_threads})

    #print("Generating grid")
    grid = bacco.visualization.uniform_grid(npix=n_grid, L=box_size, ndim=3, bounds=False)

    #print("Adding predicted displacements")
    pos = bacco.scaler.add_displacement(None,
                                        disp,
                                        box=box_size,
                                        pos=grid.reshape(-1,3),
                                        vel=None,
                                        vel_factor=0,
                                        verbose=False)[0]

    n_grid_orig = dens_lin.shape[-1]
    print(f"n_grid_orig = {n_grid_orig}")
    bmodel = bacco.BiasModel(sim=None, linear_delta=dens_lin[0], ngrid=n_grid_orig, ngrid1=None,
                            sdm=False, mode="dm", BoxSize=box_size,
                            npart_for_fake_sim=n_grid, damping_scale=damping_scale,
                            bias_model='expansion', deposit_method="cic",
                            use_displacement_of_nn=False, interlacing=interlacing,
                            )

    bias_fields = bmodel.bias_terms_lag()
    
    if velocities is not None:
        assert cosmo is not None, "must also pass cosmo for zspace!"
        pos = bacco.statistics.compute_zsd(pos, velocities, cosmo, box_size, zspace_axis=2)
    bias_terms_eul = []
    for ii in range(0,len(bias_fields)):
        bias_terms = bacco.statistics.compute_mesh(ngrid=n_grid, box=box_size, pos=pos,
                                mass = (bias_fields[ii]).flatten(), deposit_method='cic',
                                interlacing=interlacing)
        bias_terms_eul.append(bias_terms)
    bias_terms_eul = np.array(bias_terms_eul)
    
    if fn_fields is not None:
        np.save(fn_fields, bias_terms_eul)
        
    return bias_terms_eul
    

def get_tracer_field(bias_fields_eul, bias_vector, n_grid_norm=512):

    def _sum_bias_fields(fields, bias_vector):
        bias_vector_extended = np.concatenate(([1.0], bias_vector))
        return np.sum([fields[ii]*bias_vector_extended[ii] for ii in range(len(bias_vector))], axis=0)
    
    tracer_field_eul = _sum_bias_fields(bias_fields_eul, bias_vector)
    tracer_field_eul_norm = tracer_field_eul/n_grid_norm**3
    
    return tracer_field_eul_norm


def compute_pk(tracer_field, cosmo, box_size,
               k_min=0.01, k_max=1.0, n_bins=50, log_binning=True,
               normalise_grid=False, deconvolve_grid=False,
               interlacing=False, deposit_method='cic',
               correct_grid=False,
               n_threads=8, fn_pk=None):

    # NOTE by default assumes tracer field is already normalized!

    # n_grid has to match the tracer field size for this compuation!
    n_grid = tracer_field.shape[-1]
    print("Computing pk, using n_grid = ", n_grid)

    # defaults from bacco.statistics.compute_crossspectrum_twogrids
    # unless passed or otherwise denoted
    args_power_grid = {
        # "grid1": None,
        # "grid2": None,
        "normalise_grid1": normalise_grid, #default: False
        "normalise_grid2": normalise_grid, #default: False
        "deconvolve_grid1": deconvolve_grid, #default: False
        "deconvolve_grid2": deconvolve_grid, #default: False
        "ngrid": n_grid,
        "box": box_size,
        "mass1": None,
        "mass2": None,
        "interlacing": interlacing, #default: True
        "deposit_method": deposit_method, #default: "tsc",
        "log_binning": log_binning,
        "pk_lt": None,
        "kmin": k_min,
        "kmax": k_max,
        "nbins": n_bins,
        "correct_grid": correct_grid,
        "zspace": False,
        "cosmology": cosmo,
        "pmulti_interp": "polyfit",
        "nthreads": n_threads,
        "compute_correlation": False, #default: True
        "compute_power2d": False, #default: True
        "folds": 1,
        "totalmass1": None,
        "totalmass2": None,
        "jack_error": False,
        "n_jack": None
    }

    pknbody_dict = {
        'ngrid': n_grid,
        'min_k': k_min,
        'log_binning': log_binning,
        'log_binning_kmax': k_max,
        'log_binning_nbins': n_bins,
        'interlacing': interlacing,
        'depmethod': deposit_method,
        'correct_grid': correct_grid,
        'folds': 1 #default
    }
    bacco.configuration.update({'number_of_threads': n_threads})
    bacco.configuration.update({'pknbody': pknbody_dict})
    bacco.configuration.update({'pk' : {'maxk' : k_max}})
    bacco.configuration.update({'scaling' : {'disp_ngrid' : n_grid}})

    pk = bacco.statistics.compute_crossspectrum_twogrids(
                        grid1=tracer_field,
                        grid2=tracer_field,
                        **args_power_grid)
    if fn_pk is not None:
        np.save(fn_pk, pk)
    return pk


def fv2bro(t_fv_field) :
    '''Returns back row ordered array (shape n_grid * n_grid * n_grid, 3) from front vector (shape 3, n_grid, n_grid, n_grid)'''
    return np.reshape(t_fv_field, (3, int(t_fv_field.size / 3))).T



if __name__=='__main__':
    main()