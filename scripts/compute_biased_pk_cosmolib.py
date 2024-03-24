import os
os.environ["OMP_NUM_THREADS"] = str(1)

import numpy as np
from pathlib import Path
import time

import bacco


param_names = ['omega_cold', 'sigma_8', 'h', 'omega_baryon', 'n_s', 'seed']


def main():
    dir_mocks = '../data/cosmolib'
    tag_pk = '_b1000'
    dir_pks = f'../data/pks_cosmolib/pks{tag_pk}'
    overwrite = False
    
    Path.mkdir(Path(dir_pks), parents=True, exist_ok=True)
    
    bias_vector = [1., 0., 0., 0.]
    n_grid = 128
    
    fn_bias_vector = f'{dir_pks}/bias_params.txt'
    np.savetxt(fn_bias_vector, bias_vector)
    
    #n_lib = 1
    n_lib = 500
    for idx_LH in range(n_lib):
        if idx_LH%10==0:
            print(idx_LH)
        fn_fields = f'{dir_mocks}/LH{idx_LH}/Eulerian_fields_lr_{idx_LH}.npy'
        fn_params = f'{dir_mocks}/LH{idx_LH}/cosmo_{idx_LH}.txt'
        fn_pk = f'{dir_pks}/pk_{idx_LH}.npy'
        if os.path.exists(fn_pk) and not overwrite:
            print(f"P(k) for idx_LH={idx_LH} exists and overwrite={overwrite}, continuing")
            continue
        
        start = time.time()
        tracer_field = get_tracer_field(fn_fields, bias_vector)
        compute_pk(tracer_field, fn_params, n_grid, fn_pk)
        end = time.time()
        print(f"Computed P(k) for idx_LH={idx_LH} in time {end-start} s")
    
    

def get_tracer_field(fn_fields, bias_vector):

    bias_fields_eul = np.load(fn_fields)
    def _sum_bias_fields(fields, bias_vector):
        bias_vector_extended = np.concatenate(([1.0], bias_vector))
        return np.sum([fields[ii]*bias_vector_extended[ii] for ii in range(len(bias_vector))], axis=0)
    
    tracer_field_eul = _sum_bias_fields(bias_fields_eul, bias_vector)
    print(tracer_field_eul.shape)
    # normalize by 512 because that's the original ngrid size
    tracer_field_eul_norm512 = tracer_field_eul/512**3
    return tracer_field_eul_norm512


def compute_pk(tracer_field, fn_params, n_grid, fn_pk, n_threads=8):

    param_vals = np.loadtxt(fn_params)
    param_dict = dict(zip(param_names, param_vals))
    cosmo = get_cosmo(param_dict)
    
    box_size = 1000.0

    k_min = 0.01
    k_max = 0.4
    n_bins = 30
    log_binning = True

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
    
    np.save(fn_pk, pk)


def get_cosmo(param_dict):
    a_scale = 1
    cosmopars = dict(
            # careful, this is CDM! Om_cdm = Om_cold - Om_baryon
            omega_cdm=param_dict['omega_cold']-param_dict['omega_baryon'],
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