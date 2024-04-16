import os
os.environ["OMP_NUM_THREADS"] = str(1)

import numpy as np
from pathlib import Path
import time

import bacco



def main():
    compute_pks_muchisimocks()
    #compute_pks_quijote_LH()
    

def compute_pks_muchisimocks():
    tag_mocks = '_HR'
    dir_mocks = f'../data/cosmolib{tag_mocks}'
    tag_pk = '_b0000'
    dir_pks = f'../data/pks_cosmolib/pks{tag_mocks}{tag_pk}'
    tag_fields = '_hr'
    #tag_fields = '_lr'
    tag_fields_extra = '_2GpcBox'
    overwrite = True
    
    Path.mkdir(Path(dir_pks), parents=True, exist_ok=True)

    bias_vector = [0., 0., 0., 0.]    
    fn_bias_vector = f'{dir_pks}/bias_params.txt'
    np.savetxt(fn_bias_vector, bias_vector)

    #n_grid = 128
    n_grid_orig = None #compute from fields, if we don't know that it's different
    #n_grid_orig = 512
    if '2Gpc' in tag_fields_extra:
        box_size = 2000.0
    else:
        box_size = 1000.0
    
    # order of saved cosmo param files
    param_names = ['omega_cold', 'sigma_8', 'h', 'omega_baryon', 'n_s', 'seed']

    n_lib = 1
    #n_lib = 500
    for idx_LH in range(n_lib):
        if idx_LH%10==0:
            print(idx_LH)
        fn_fields = f'{dir_mocks}/LH{idx_LH}/Eulerian_fields{tag_fields}_{idx_LH}{tag_fields_extra}.npy'
        fn_params = f'{dir_mocks}/LH{idx_LH}/cosmo_{idx_LH}.txt'
        fn_pk = f'{dir_pks}/pk_{idx_LH}{tag_fields_extra}.npy'
        if os.path.exists(fn_pk) and not overwrite:
            print(f"P(k) for idx_LH={idx_LH} exists and overwrite={overwrite}, continuing")
            continue
        
        start = time.time()
        # normalize by 512 because that's the original ngrid size
        bias_terms_eul = np.load(fn_fields)
        if n_grid_orig is None:
            n_grid_orig = bias_terms_eul.shape[-1]
        print(f"n_grid_orig = {n_grid_orig}")
        tracer_field = get_tracer_field(bias_terms_eul, bias_vector, n_grid_norm=n_grid_orig)
        
        param_vals = np.loadtxt(fn_params)
        param_dict = dict(zip(param_names, param_vals))
        cosmo = get_cosmo(param_dict)
        
        compute_pk(tracer_field, cosmo, box_size, fn_pk=fn_pk)
        end = time.time()
        print(f"Computed P(k) for idx_LH={idx_LH} in time {end-start} s")
    

# TODO: split into 2 functions, first gen and save bias fields, then separate to compute pks
def compute_pks_quijote_LH():
    dir_mocks = '/cosmos_storage/home/mpelle/Yin_data/Quijote'
    dir_fields = '../data/quijote_LH'
    tag_pk = '_b0000'
    dir_pks_sim = f'../data/pks_quijote_LH/pks_sim{tag_pk}'
    dir_pks_pred = f'../data/pks_quijote_LH/pks_pred{tag_pk}'
    overwrite = False
    
    Path.mkdir(Path(dir_pks_sim), parents=True, exist_ok=True)
    Path.mkdir(Path(dir_pks_pred), parents=True, exist_ok=True)
    
    bias_vector = [0., 0., 0., 0.]
    n_grid = 512
    n_grid_orig = 512
    box_size = 1000.0
    # order of saved cosmo param files (via https://quijote-simulations.readthedocs.io/en/latest/LH.html)
    # careful that here it's omega_m, whereas for muchisimocks/cosmolib it's omega_cold
    # (should be handled in get_cosmo() function)
    param_names = ['omega_m', 'omega_baryon', 'h', 'n_s', 'sigma_8']

    fn_bias_vector = f'{dir_pks_sim}/bias_params.txt'
    np.savetxt(fn_bias_vector, bias_vector)
    fn_bias_vector = f'{dir_pks_pred}/bias_params.txt'
    np.savetxt(fn_bias_vector, bias_vector)
    
    idxs_LH = np.array([10,29,37,40,70,85,127,158,165,184,208,220,240,254,267,274,293,305,336,374,375,388,433,444,
                      464,502,534,542,574,598,605,628,652,663,676,700,702,721,737,762,809,822,825,837,853,864,882,
                      899,901,911,939,948,950,951,964,976,977,1016,1022,1041,1050,1060,1082,1091,1103,1114,1147,
                      1157,1173,1175,1219,1222,1299,1309,1314,1317,1331,1365,1372,1378,1391,1397,1418,1444,1459,
                      1510,1512,1513,1515,1517,1533,1553,1567,1568,1599,1622,1642,1657,1659,1667])

    for idx_LH in idxs_LH:
    #for idx_LH in idxs_LH[:1]:
        if idx_LH%10==0:
            print(idx_LH)
        idx_LH_str = f'{idx_LH:04}'
        
        # SIMULATED
        fn_pk_sim = f'{dir_pks_sim}/pk_{idx_LH_str}.npy'
        if os.path.exists(fn_pk_sim) and not overwrite:
            print(f"P(k) for orig sim for idx_LH={idx_LH} exists and overwrite={overwrite}, continuing")
            continue
    
        # set up needed for both sim and pred 
        k_nyq = np.pi * n_grid / box_size
        damping_scale = k_nyq
        k_min = 0.01
        k_max = 1
        n_bins = 50
        fn_dens_lin = f'{dir_mocks}/LH{idx_LH_str}/lin_den_{idx_LH_str}.npy'
        dens_lin = np.load(fn_dens_lin)

        # get params
        fn_params = f'{dir_mocks}/LH{idx_LH_str}/param_{idx_LH_str}.txt'
        param_vals = np.loadtxt(fn_params)
        param_dict = dict(zip(param_names, param_vals))
        cosmo = get_cosmo(param_dict)
        
        # get fields sim
        fn_disp_sim = f'{dir_mocks}/LH{idx_LH_str}/dis_{idx_LH_str}.npy'
        fn_fields_sim = f'{dir_fields}/LH{idx_LH_str}/Eulerian_fields_sim_{idx_LH_str}.npy'
        Path.mkdir(Path(f'{dir_fields}/LH{idx_LH_str}'), parents=True, exist_ok=True)
    
        disp_sim = np.load(fn_disp_sim)
        start = time.time()
        bias_terms_eul_sim = displacements_to_bias_fields(dens_lin, disp_sim, n_grid, box_size, 
                                    damping_scale=damping_scale, fn_fields=fn_fields_sim)
        end = time.time()
        print(f"Generated bias fields for orig sim for idx_LH={idx_LH} in time {end-start} s")
            
        # compute sim pk
        start = time.time()
        tracer_field_sim = get_tracer_field(bias_terms_eul_sim, bias_vector, n_grid_norm=n_grid_orig)
        compute_pk(tracer_field_sim, cosmo, box_size,
                   k_min=k_min, k_max=k_max, n_bins=n_bins,
                   fn_pk=fn_pk_sim)
        end = time.time()
        print(f"Computed P(k) for orig sim for idx_LH={idx_LH} in time {end-start} s")

        # PREDICTED
        fn_pk_pred = f'{dir_pks_pred}/pk_{idx_LH_str}.npy'
        if os.path.exists(fn_pk_pred) and not overwrite:
            print(f"P(k) for map2map prediction for idx_LH={idx_LH} exists and overwrite={overwrite}, continuing")
            continue

        # get fields map2map pred
        fn_disp_pred = f'{dir_mocks}/LH{idx_LH_str}/pred_pos_{idx_LH_str}.npy'
        fn_fields_pred = f'{dir_fields}/LH{idx_LH_str}/Eulerian_fields_pred_{idx_LH_str}.npy'
        Path.mkdir(Path(f'{dir_fields}/LH{idx_LH_str}'), parents=True, exist_ok=True)

        disp_pred = np.load(fn_disp_pred)
        start = time.time()
        bias_terms_eul_pred = displacements_to_bias_fields(dens_lin, disp_pred, n_grid, box_size, 
                                    damping_scale=damping_scale, fn_fields=fn_fields_pred)
        end = time.time()
        print(f"Generated bias fields for map2map pred for idx_LH={idx_LH} in time {end-start} s")
         
        # compute map2map pred pk
        start = time.time()
        tracer_field_pred = get_tracer_field(bias_terms_eul_pred, bias_vector, n_grid_norm=n_grid_orig)
        compute_pk(tracer_field_pred, cosmo, box_size,
                   k_min=k_min, k_max=k_max, n_bins=n_bins,
                   fn_pk=fn_pk_pred)
        end = time.time()
        print(f"Computed P(k) for map2map pred for idx_LH={idx_LH} in time {end-start} s")
    
    
def displacements_to_bias_fields(dens_lin, disp, n_grid, box_size, 
                                 damping_scale=None, n_threads=8,
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

    bmodel = bacco.BiasModel(sim=None, linear_delta=dens_lin[0], ngrid=n_grid, ngrid1=None,
                            sdm=False, mode="dm", BoxSize=box_size,
                            npart_for_fake_sim=n_grid, damping_scale=damping_scale,
                            bias_model='expansion', deposit_method="cic",
                            use_displacement_of_nn=False, interlacing=False,
                            )

    bias_fields = bmodel.bias_terms_lag()

    bias_terms_eul = []
    for ii in range(0,len(bias_fields)):
        bias_terms = bacco.statistics.compute_mesh(ngrid=n_grid, box=box_size, pos=pos,
                                mass = (bias_fields[ii]).flatten(), deposit_method='cic',
                                interlacing=False)
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
               k_min=0.01, k_max=0.4, n_bins=30, log_binning=True,
               n_threads=8, fn_pk=None):

    # n_grid has to match the tracer field size!
    n_grid = tracer_field.shape[-1]
    print("To compute pk, using n_grid = ", n_grid)
    
    args_power = {'ngrid':n_grid,
                'box':box_size,
                'cosmology':cosmo,
                'interlacing':False,
                'kmin':k_min,
                'kmax':k_max,
                'nbins':n_bins,
                'correct_grid':True,
                'log_binning':log_binning,
                'deposit_method':'cic',
                'compute_correlation':False,
                'zspace':False,
                'compute_power2d':False}
    
    bacco.configuration.update({'number_of_threads': n_threads})

    # NOTE assumes tracer field is already normalized!
    pk = bacco.statistics.compute_crossspectrum_twogrids(
                        grid1=tracer_field,
                        grid2=tracer_field,
                        normalise_grid1=False,
                        normalise_grid2=False,
                        deconvolve_grid1=False,
                        deconvolve_grid2=False,
                        **args_power)
    if fn_pk is not None:
        np.save(fn_pk, pk)
    return pk

def get_cosmo(param_dict):
    a_scale = 1
    # omega_m = omega_cold + omega_neutrinos 
    # (omega_m = omega_cold if no neutrinos) 
    # Om_cdm = Om_cold - Om_baryon
    if 'omega_m' in param_dict:
        omega_cdm = param_dict['omega_m']
    elif 'omega_cold' in param_dict:
        omega_cdm = param_dict['omega_cold']-param_dict['omega_baryon']
    else:
        raise ValueError("param_dict must include omega_m or omega_cold!")

    cosmopars = dict(
            omega_cdm=omega_cdm,
            omega_baryon=param_dict['omega_baryon'],
            hubble=param_dict['h'],
            ns=param_dict['n_s'],
            sigma8=param_dict['sigma_8'],
            tau=0.0561,
            A_s=None,
            neutrino_mass=0.,
            w0=-1,
            wa=0,
        )

    cosmo = bacco.Cosmology(**cosmopars)
    cosmo.set_expfactor(a_scale)
    return cosmo




if __name__=='__main__':
    main()