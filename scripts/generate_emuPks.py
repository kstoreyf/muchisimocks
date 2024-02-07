import numpy as np
from scipy.stats import qmc

import baccoemu


def main():

    tag_emuPk = '_5param'

    param_bounds = {'omega_cold': [0.23, 0.4],
                    'sigma8_cold': [0.73, 0.9],
                    'hubble': [0.6, 0.8],
                    'ns': [0.92, 1.01],
                    'omega_baryon': [0.04, 0.06]}
    #param_names = ['omega_cold', 'sigma8_cold']
    param_names = ['omega_cold', 'sigma8_cold', 'hubble', 'ns', 'omega_baryon']
    n_params = len(param_names)
    n_tot = 1000
    
    sampler = qmc.LatinHypercube(d=n_params, seed=42)
    theta_orig = sampler.random(n=n_tot)

    l_bounds = [param_bounds[param_name][0] for param_name in param_names]
    u_bounds = [param_bounds[param_name][1] for param_name in param_names]
    theta = qmc.scale(theta_orig, l_bounds, u_bounds)

    emu = baccoemu.Lbias_expansion()
    cosmo_params = setup_cosmo_emu()
    bias_params = [1., 0., 0., 0.]
    k = np.logspace(-2, np.log10(0.75), 30)

    pks = []
    for i in range(len(theta)):
        for pp in range(len(param_names)):
            cosmo_params[param_names[pp]] = theta[i][pp]
        _, pk_gg, _ = emu.get_galaxy_real_pk(bias=bias_params, k=k, 
                                                    **cosmo_params)
        pks.append(pk_gg)
        
    fn_emuPk = f'../data/emuPks/emuPks{tag_emuPk}.npy'
    fn_emuPk_params = f'../data/emuPks/emuPks_params{tag_emuPk}.txt'
    fn_emuk = f'../data/emuPks/emuPks_k{tag_emuPk}.txt'

    np.save(fn_emuPk, pks)
    header = ','.join(param_names)
    np.savetxt(fn_emuPk_params, theta, header=header, delimiter=',', fmt='%.8f')
    np.savetxt(fn_emuk, k)
           



def setup_cosmo_emu():
    print("Setting up emulator cosmology")
    Ob = 0.049
    Om = 0.3175
    hubble = 0.6711
    ns = 0.9624
    sigma8 = 0.834
    cosmo_params = {
        'omega_cold'    :  Om,
        'sigma8_cold'   :  sigma8, # if A_s is not specified
        'omega_baryon'  :  Ob,
        'ns'            :  ns,
        'hubble'        :  hubble,
        'neutrino_mass' :  0.0,
        'w0'            : -1.0,
        'wa'            :  0.0,
        'expfactor'     :  1
    }
    return cosmo_params



if __name__=='__main__':
    main()