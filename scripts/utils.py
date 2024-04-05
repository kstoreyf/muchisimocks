import getdist
import numpy as np 
import os

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



def load_emu(emu_name='2.0'):
    if emu_name=='public':
        emu = baccoemu.Lbias_expansion(verbose=False)
    elif emu_name=='2.0':
        fn_emu = '/cosmos_storage/cosmosims/data_share/lbias_emulator/lbias_emulator2.0.0'
        emu = baccoemu.Lbias_expansion(verbose=False, 
                                    nonlinear_emu_path=fn_emu,
                                    nonlinear_emu_details='details.pickle',
                                    nonlinear_emu_field_name='NN_n',
                                    nonlinear_emu_read_rotation=False)
    else:
        raise ValueError(f'Emulator {emu_name} not recognized!')
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


def get_samples(idx_obs, inf_method, tag_inf):
    if inf_method == 'mn':
        return get_samples_mn(idx_obs, tag_inf)
    elif inf_method == 'emcee':
        return get_samples_emcee(idx_obs, tag_inf)
    elif inf_method == 'dynesty':
        return get_samples_dynesty(idx_obs, tag_inf)
    else:
        raise ValueError(f'Method {inf_method} not recognized!')
        
        
def get_moments_test_mn(tag_inf):
    dir_mn = f'../data/results_moment_network/mn{tag_inf}'
    theta_test_pred = np.load(f'{dir_mn}/theta_test_pred.npy')
    covs_test_pred = np.load(f'{dir_mn}/covs_test_pred.npy')
    return theta_test_pred, covs_test_pred
    
        
def get_samples_mn(idx_obs, tag_inf):
    rng = np.random.default_rng(42)
    dir_mn = f'../data/results_moment_network/mn{tag_inf}'
    theta_test_pred = np.load(f'{dir_mn}/theta_test_pred.npy')
    covs_test_pred = np.load(f'{dir_mn}/covs_test_pred.npy')
    try:
        samples = rng.multivariate_normal(theta_test_pred[idx_obs], 
                                            covs_test_pred[idx_obs], int(1e6),
                                            check_valid='raise')
    except ValueError:
        title += f' [$C$ not PSD!]'
        samples = rng.multivariate_normal(theta_test_pred[idx_obs], 
                                            covs_test_pred[idx_obs], int(1e6),
                                            check_valid='ignore')
    return samples


def get_samples_emcee(idx_obs, tag_inf):
    import emcee
    dir_emcee =  f'../data/results_emcee/samplers{tag_inf}'
    fn_emcee = f'{dir_emcee}/sampler_idxtest{idx_obs}.npy'
    if not os.path.exists(fn_emcee):
        print(f'File {fn_emcee} not found')
        return
    reader = emcee.backends.HDFBackend(fn_emcee)

    tau = reader.get_autocorr_time()
    n_burn = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    #print(n_burn, thin)
    samples = reader.get_chain(discard=n_burn, flat=True, thin=thin)
    return samples


def get_samples_dynesty(idx_obs, tag_inf):
    dir_dynesty =  f'../data/results_dynesty/samplers{tag_inf}'
    fn_dynesty = f'{dir_dynesty}/sampler_results_idxtest{idx_obs}.npy'
    results_dynesty = np.load(fn_dynesty, allow_pickle=True).item()
    
    # doesn't work upon reload for some reason
    #samples_dynesty = results_dynesty.samples_equal()
    
    from dynesty.utils import resample_equal
    # draw posterior samples
    weights = np.exp(results_dynesty['logwt'] - results_dynesty['logz'][-1])
    samples = resample_equal(results_dynesty.samples, weights)

    return samples


def repeat_arr_rlzs(arr, n_rlzs=1):
    arr_repeat = np.repeat(arr, n_rlzs, axis=0)
    return arr_repeat