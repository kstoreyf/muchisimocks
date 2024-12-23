import numpy as np
import os
from scipy.stats import qmc

import bacco

import utils


def main():

    n_emuPk = 1000
    #tag_emuPk = f'_5param_n{n_emuPk}'
    #tag_emuPk = f'_2param_n{n_emuPk}'
    tag_emuPk = f'_fixedcosmo_n{n_emuPk}'

    box_size = 500.
    tag_errG = f'_boxsize{int(box_size)}'
    n_rlzs_per_cosmo = 1
    tag_datagen = f'{tag_emuPk}{tag_errG}_nrlzs{n_rlzs_per_cosmo}'

    fn_emuPk = f'../data/emuPks/emuPks{tag_emuPk}.npy'
    fn_emuPkerrG = f'../data/emuPks/emuPks_errgaussian{tag_emuPk}{tag_errG}.npy'
    fn_emuPk_noisy = f'../data/emuPks/emuPks_noisy{tag_datagen}.npy'
    fn_emuPk_params = f'../data/emuPks/emuPks_params{tag_emuPk}.txt'
    fn_emuk = f'../data/emuPks/emuPks_k{tag_emuPk}.txt'
    fn_bias_vector = f'../data/emuPks/bias_params.txt'
    fn_rands = f'../data/emuPks/randints{tag_emuPk}.npy'

    bias_params = [1., 0., 0., 0.]
    np.savetxt(fn_bias_vector, bias_params)
    
    
    dir_emus_lbias = '/home/kstoreyf/external'
    emu, emu_bounds, emu_param_names = utils.load_emu(dir_emus_lbias=dir_emus_lbias)
    
    if '_2param' in tag_datagen:
        param_names = ['omega_cold', 'sigma8_cold']
    elif '_5param' in tag_datagen:
        param_names = ['omega_cold', 'sigma8_cold', 'hubble', 'ns', 'omega_baryon']
    elif '_fixedcosmo' in tag_datagen:
        param_dict = utils.cosmo_dict_quijote
        param_names = param_dict.keys()
        param_names = [pn for pn in param_names if pn in emu_param_names]
    else:
        raise KeyError('define parameters!')
    print(param_names)

    param_bounds = {name: emu_bounds[emu_param_names.index(name)] for name in param_names}
    
    if '_fixedcosmo' in tag_datagen:
        theta_single = np.array([param_dict[pn] for pn in param_names])
        theta = np.tile(theta_single, (n_emuPk, 1))
        header = ','.join(param_names)
        # redundant to save all of them but doing it for consistency
        np.savetxt(fn_emuPk_params, theta, header=header, delimiter=',', fmt='%.8f')
    else:
        theta = latin_hypercube(param_names, param_bounds, n_emuPk,
                                fn_emuPk_params=fn_emuPk_params)
    print(theta.shape)
    k, Pk_noiseless = generate_pks(theta, bias_params, param_names, emu, 
                         fn_emuk=fn_emuk, fn_emuPk=fn_emuPk)
    gaussian_error_pk = compute_noise(k, Pk_noiseless, box_size, fn_emuPkerrG=fn_emuPkerrG)
    draw_noisy_pk_realizations(Pk_noiseless, theta, gaussian_error_pk, 
                               n_rlzs_per_cosmo=n_rlzs_per_cosmo,
                               fn_emuPk_noisy=fn_emuPk_noisy)
    utils.generate_randints(n_emuPk, fn_rands)

    
def generate_pks(theta, bias_params, param_names, emu,
                 fn_emuk=None, fn_emuPk=None, overwrite=False):
    
    if os.path.exists(fn_emuk) and os.path.exists(fn_emuPk) and not overwrite:
        print(f"Loading from {fn_emuk} and {fn_emuPk} (already exist)")
        k = np.genfromtxt(fn_emuk)
        Pk = np.load(fn_emuPk)
        return k, Pk
    
    print(f"Generating emuPks for {fn_emuPk}...")
    cosmo_params = utils.setup_cosmo_emu(cosmo='quijote')
    cosmo_params['expfactor'] = 1
    k = np.logspace(-2, np.log10(0.75), 30)

    Pk = []
    for i in range(len(theta)):
        for pp in range(len(param_names)):
            cosmo_params[param_names[pp]] = theta[i][pp]
        _, pk_gg, _ = emu.get_galaxy_real_pk(bias=bias_params, k=k, 
                                                    **cosmo_params)
        Pk.append(pk_gg)

    np.save(fn_emuPk, Pk)
    np.savetxt(fn_emuk, k)
    return np.array(k), np.array(Pk)
           
  
def latin_hypercube(param_names, param_bounds, n_tot, 
                    fn_emuPk_params=None, overwrite=False):
    
    if os.path.exists(fn_emuPk_params) and not overwrite:  
        print(f"Loading from {fn_emuPk_params} (already exists)")
        theta = np.genfromtxt(fn_emuPk_params, delimiter=',', names=True)
        return theta 
    
    print(f"Generating latin hypercube for {fn_emuPk_params}...")
    n_params = len(param_names)
    sampler = qmc.LatinHypercube(d=n_params, seed=42)
    theta_orig = sampler.random(n=n_tot)

    l_bounds = [param_bounds[param_name][0] for param_name in param_names]
    u_bounds = [param_bounds[param_name][1] for param_name in param_names]
    theta = qmc.scale(theta_orig, l_bounds, u_bounds)
            
    header = ','.join(param_names)
    np.savetxt(fn_emuPk_params, theta, header=header, delimiter=',', fmt='%.8f')

    return theta


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


def draw_noisy_pk_realizations(Pk_noiseless, theta, gaussian_error_pk, n_rlzs_per_cosmo=1,
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
