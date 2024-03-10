import numpy as np
from scipy.stats import qmc

import baccoemu


def main():

    tag_emuPk = '_2param'
    n_tot = 10000

    #emu = baccoemu.Lbias_expansion(verbose=False)
    fn_emu = '/dipc_storage/cosmosims/data_share/lbias_emulator/lbias_emulator2.0.0'
    emu = baccoemu.Lbias_expansion(nonlinear_emu_path=fn_emu,
                                nonlinear_emu_details='details.pickle',
                                nonlinear_emu_field_name='NN_n',
                                nonlinear_emu_read_rotation=False,
                                verbose=False)
    
    if '_2param' in tag_emuPk:
        param_names = ['omega_cold', 'sigma8_cold']
    elif '_5param' in tag_emuPk:
        param_names = ['omega_cold', 'sigma8_cold', 'hubble', 'ns', 'omega_baryon']
    else:
        raise KeyError('define parameters!')
    
    param_keys = emu.emulator['nonlinear']['keys']
    emu_bounds =  emu.emulator['nonlinear']['bounds']
    param_bounds = {name: emu_bounds[param_keys.index(name)] for name in param_names}
    
    theta = latin_hypercube(param_names, param_bounds, n_tot)
    generate_pks(theta, param_names, emu, tag_emuPk)

    
def generate_pks(theta, param_names, emu, tag_emuPk):
    
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
    return pks
           
  
def latin_hypercube(param_names, param_bounds, n_tot):
    n_params = len(param_names)
    sampler = qmc.LatinHypercube(d=n_params, seed=42)
    theta_orig = sampler.random(n=n_tot)

    l_bounds = [param_bounds[param_name][0] for param_name in param_names]
    u_bounds = [param_bounds[param_name][1] for param_name in param_names]
    theta = qmc.scale(theta_orig, l_bounds, u_bounds)
    return theta


def setup_cosmo_emu():
    print("Setting up emulator cosmology")
    cosmo_params = {
        'omega_cold'    :  0.3175,
        'sigma8_cold'   :  0.834, # if A_s is not specified
        'omega_baryon'  :  0.049,
        'ns'            :  0.9624,
        'hubble'        :  0.6711,
        'neutrino_mass' :  0.0,
        'w0'            : -1.0,
        'wa'            :  0.0,
        'expfactor'     :  1.0
    }
    return cosmo_params





if __name__=='__main__':
    main()
