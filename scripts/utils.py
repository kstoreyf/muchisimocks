import getdist
import numpy as np 
import os

import baccoemu


param_label_dict = {'omega_cold': r'$\Omega_\mathrm{cold}$',
                'sigma8_cold': r'$\sigma_{8}$',
                'sigma_8': r'$\sigma_{8}$',
                'hubble': r'$h$',
                'h': r'$h$',
                'ns': r'$n_\mathrm{s}$',
                'n_s': r'$n_\mathrm{s}$',
                'omega_baryon': r'$\Omega_\mathrm{b}$',
                'omega_m': r'$\Omega_\mathrm{m}$',
                }

color_dict_methods = {'mn': 'blue',
                      'emcee': 'purple',
                      'dynesty': 'red'}

label_dict_methods = {'mn': 'Moment Network',
                      'emcee': 'MCMC (emcee)',
                      'dynesty': 'MCMC (dynesty)'}

labels_pnn = ['$1 1$',
            '$1 \\delta$',
            '$1 \\delta^2$',
            '$1 s^2$',
            '$ 1 \\nabla^2\\delta$',
            '$\\delta \\delta$',
            '$\\delta \\delta^2$',
            '$\\delta s^2$',
            '$\\delta \\nabla^2\\delta$',
            '$\\delta^2 \\delta^2$',
            '$\\delta^2 s^2$',
            '$\\delta^2 \\nabla^2\\delta$',
            '$s^2 s^2$',
            '$s^2 \\nabla^2\\delta$',
            '$\\nabla^2\\delta \\nabla^2\\delta$'
            ]

# https://arxiv.org/pdf/1909.05273, Table 1, top row
cosmo_dict_quijote = {
                'omega_cold'    :  0.3175,
                'omega_baryon'  :  0.049,
                'sigma8_cold'   :  0.834,
                'ns'            :  0.9624,
                'hubble'        :  0.6711,
                'neutrino_mass' :  0.0,
                'w0'            : -1.0,
                'wa'            :  0.0,
                'tau'           :  0.0561, #planck value
                }   

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
        cosmo_params = cosmo_dict_quijote
    else:
        raise ValueError(f'Cosmo {cosmo} not recognized!')
    return cosmo_params


def load_emu(emu_name='lbias_2.0'):
    dir_emus_lbias = '/cosmos_storage/data_sharing/data_share'
    dir_emus_mpk = '/cosmos_storage/data_sharing/datashare'
    if emu_name=='lbias_public':
        emu = baccoemu.Lbias_expansion(verbose=False)
    elif emu_name=='lbias_2.0':
        fn_emu = f'{dir_emus_lbias}/lbias_emulator/lbias_emulator2.0.0'
        emu = baccoemu.Lbias_expansion(verbose=False, 
                                    nonlinear_emu_path=fn_emu,
                                    nonlinear_emu_details='details.pickle',
                                    nonlinear_emu_field_name='NN_n',
                                    nonlinear_emu_read_rotation=False)
    elif emu_name=='mpk':
        standardspace_folder = f'{dir_emus_mpk}/mpk_baccoemu_new/mpk_oldsims_standard_emu_npca7_neurons_400_400_dropout_0.0_bn_False/'
        emu = baccoemu.Matter_powerspectrum(nonlinear_emu_path=standardspace_folder, 
                                                     nonlinear_emu_details='details.pickle',
                                                     verbose=False,)
    elif emu_name=='mpk_extended':
        extendedspace_folder = f'{dir_emus_mpk}/mpk_baccoemu_new/mpk_extended_emu_npca_20_batch_size_256_nodes_400_400_dropout_0.0_batch_norm_False/'
        emu = baccoemu.Matter_powerspectrum(nonlinear_emu_path=extendedspace_folder, 
                                            nonlinear_emu_details='details.pickle',
                                            verbose=False,)
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
    arr_repeat = np.tile(arr, (n_rlzs,1))
    return arr_repeat


def param_dict_to_bacco_param_dict(param_dict, neutrino_mass=None):
    
    if neutrino_mass is None:
        assert 'neutrino_mass' in param_dict, "must pass neutrino mass in param dict or separately!"
        neutrino_mass = param_dict['neutrino_mass']
        
    names_to_bacco = {'h': 'hubble',
                      'sigma_8': 'sigma8',
                      'sigma8_cold': 'sigma8',
                      'n_s': 'ns'}
    param_dict_bacco = {}
    for name_orig in param_dict:
        if name_orig in names_to_bacco:
            name_bacco = names_to_bacco[name_orig]
        else:
            name_bacco = name_orig
        param_dict_bacco[name_bacco] = param_dict[name_orig] 
        
    pn_oms = ['omega_m', 'omega_cdm', 'omega_cold']
    n_oms = np.sum([1 if pn_om in param_dict_bacco else 0 for pn_om in pn_oms])        
    assert n_oms==1, f"should pass exactly one of {pn_oms}!"     
              
    if 'omega_cdm' in param_dict_bacco:
        pass      
    elif 'omega_cold' in param_dict_bacco:                   
        param_dict_bacco['omega_cdm'] = param_dict_bacco['omega_cold']-param_dict_bacco['omega_baryon']
    elif 'omega_m' in param_dict_bacco:
        param_dict_bacco['omega_cold'] = param_dict_bacco['omega_m'] - neutrino_mass
        param_dict_bacco['omega_cdm'] = param_dict_bacco['omega_cold']-param_dict_bacco['omega_baryon']
        
    return param_dict_bacco


def get_cosmo(param_dict, a_scale=1, sim_name='quijote'):
    import bacco
    
    param_names_bacco = ['omega_cdm', 'omega_baryon', 'hubble', 'ns', 'sigma8', 
                        'tau', 'neutrino_mass', 'w0', 'wa']

    if sim_name=='quijote':
        param_dict_bacco_fid = param_dict_to_bacco_param_dict(cosmo_dict_quijote)
    else:
        raise ValueError(f'Simulation {sim_name} not recognized!')

    # omega_m = omega_cold + omega_neutrinos 
    # (omega_m = omega_cold if no neutrinos) 
    # Om_cdm = Om_cold - Om_baryon
                                   
    # Ωm = Ωcdm + Ωb + Ων
    # (Ωcold = Ωcdm + Ωb)
    # (Ωm = Ωcold + Ων)              
    # Ωcdm = Ωcold - Ωb
    neutrino_mass = param_dict['neutrino_mass'] if 'neutrino_mass' in param_dict \
                                                else param_dict_bacco_fid['neutrino_mass']     

    param_dict_bacco = param_dict_to_bacco_param_dict(param_dict, neutrino_mass=neutrino_mass)

    cosmopars = {}
    for pn in param_names_bacco:
        if pn in param_dict_bacco:
            cosmopars[pn] = param_dict_bacco[pn]
        else:
            print(f"Param {pn} not in param dict, adding {sim_name} value")
            cosmopars[pn] = param_dict_bacco_fid[pn]

    cosmo = bacco.Cosmology(**cosmopars, verbose=False)
    cosmo.set_expfactor(a_scale)
    return cosmo


def get_cosmo_emu(param_dict, a_scale=1, sim_name='quijote'):
    cosmo = get_cosmo(param_dict, a_scale=a_scale, sim_name=sim_name)
    return cosmo_bacco_to_cosmo_baccoemu(cosmo)


def cosmo_bacco_to_cosmo_baccoemu(cosmo):
    
    param_names_emu = ['sigma8_cold', 'omega_cold', 'hubble', 'ns', 'omega_baryon', 
                       'expfactor', 'neutrino_mass', 'w0', 'wa']
    cosmo_params_emu = {}
    for param_name_emu in param_names_emu:
        if param_name_emu=='sigma8_cold':
            param_bacco = cosmo.pars['sigma8']
        elif param_name_emu=='expfactor':
            param_bacco = cosmo.expfactor
        else:
            param_bacco = cosmo.pars[param_name_emu]
        cosmo_params_emu[param_name_emu] = param_bacco
        
    return cosmo_params_emu


def get_tracer_field(bias_fields_eul, bias_vector, n_grid_norm=None):

    assert len(bias_vector)==bias_fields_eul.shape[0]-1, "bias_vector must length one less than number of bias fields"
    if n_grid_norm is None:
        n_grid_norm = bias_fields_eul.shape[-1]
        
    def _sum_bias_fields(fields, bias_vector):
        bias_vector_extended = np.concatenate(([1.0], bias_vector))
        return np.sum([fields[ii]*bias_vector_extended[ii] for ii in range(len(bias_vector_extended))], axis=0)
    
    tracer_field_eul = _sum_bias_fields(bias_fields_eul, bias_vector)
    tracer_field_eul_norm = tracer_field_eul/n_grid_norm**3
    
    return tracer_field_eul_norm