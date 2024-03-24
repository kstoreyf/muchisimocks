import getdist
import numpy as np 

import baccoemu

param_label_dict = {'omega_cold': r'$\Omega_\mathrm{m}$',
                'sigma8_cold': r'$\sigma_{8}$',
                'sigma_8': r'$\sigma_{8}$',
                'hubble': r'$h$',
                'h': r'$h$',
                'ns': r'$n_\mathrm{s}$',
                'n_s': r'$n_\mathrm{s}$',
                'omega_baryon': r'$\Omega_\mathrm{b}$',}

color_dict_methods = {'mn': 'blue',
                      'emcee': 'purple',
                      'dynesty': 'red'}

label_dict_methods = {'mn': 'Moment Network',
                      'emcee': 'MCMC (emcee)',
                      'dynesty': 'MCMC (dynesty)'}


def idxs_train_val_test(random_ints, frac_train=0.70, frac_val=0.15, frac_test=0.15):
    print(frac_train, frac_val, frac_test)
    tol = 1e-6
    assert abs((frac_train+frac_val+frac_test) - 1.0) < tol, "Fractions must add to 1!" 
    N_tot = len(random_ints)
    int_train = int(frac_train*N_tot)
    int_test = int((1-frac_test)*N_tot)

    idxs_train = np.where(random_ints < int_train)[0]
    idxs_test = np.where(random_ints >= int_test)[0]
    idxs_val = np.where((random_ints >= int_train) & (random_ints < int_test))[0]

    return idxs_train, idxs_val, idxs_test


def split_train_val_test(arr, idxs_train, idxs_val, idxs_test):
    arr_train = arr[idxs_train]
    arr_val = arr[idxs_val]
    arr_test = arr[idxs_test]
    return arr_train, arr_val, arr_test


def setup_cosmo_emu(cosmo='quijote'):
    print("Setting up emulator cosmology")
    if cosmo=='quijote':
        cosmo_params = {
            'omega_cold'    :  0.3175,
            'sigma8_cold'   :  0.834,
            'omega_baryon'  :  0.049,
            'ns'            :  0.9624,
            'hubble'        :  0.6711,
            'neutrino_mass' :  0.0,
            'w0'            : -1.0,
            'wa'            :  0.0,
            'expfactor'     :  1.0
        }
    else:
        raise ValueError(f'Cosmo {cosmo} not recognized!')
    return cosmo_params



def load_emu():
    #emu = baccoemu.Lbias_expansion(verbose=False)
    fn_emu = '/dipc_storage/cosmosims/data_share/lbias_emulator/lbias_emulator2.0.0'
    emu = baccoemu.Lbias_expansion(verbose=False, 
                                nonlinear_emu_path=fn_emu,
                                nonlinear_emu_details='details.pickle',
                                nonlinear_emu_field_name='NN_n',
                                nonlinear_emu_read_rotation=False)
    emu_param_names = emu.emulator['nonlinear']['keys']
    emu_bounds =  emu.emulator['nonlinear']['bounds']
    return emu, emu_bounds, emu_param_names



def get_posterior_maxes(samples_equal, param_names):
    samps = getdist.MCSamples(names=param_names)
    samps.setSamples(samples_equal)
    maxes = []
    for i, pn in enumerate(param_names):
        xvals = np.linspace(min(samples_equal[:,i]), max(samples_equal[:,i]), 1000)
        dens = samps.get1DDensity(pn)   
        probs = dens(xvals)
        posterior_max = xvals[np.argmax(probs)]
        maxes.append(posterior_max)
    return maxes