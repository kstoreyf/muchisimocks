import numpy as np 
import os



param_label_dict = {'omega_cold': r'$\Omega_\mathrm{cold}$',
                'sigma8_cold': r'$\sigma_{8}$',
                'sigma_8': r'$\sigma_{8}$',
                'hubble': r'$h$',
                'h': r'$h$',
                'ns': r'$n_\mathrm{s}$',
                'n_s': r'$n_\mathrm{s}$',
                'omega_baryon': r'$\Omega_\mathrm{b}$',
                'omega_m': r'$\Omega_\mathrm{m}$',
                'b1': r'$b_1$',
                'b2': r'$b_2$',
                'bs2': r'$b_{s^2}$',
                'bl': r'$b_{\Delta}$',
                }

color_dict_methods = {'mn': 'blue',
                      'sbi': 'green',
                      'emcee': 'purple',
                      'dynesty': 'red'}

label_dict_methods = {'mn': 'Moment Network',
                      'sbi': 'SBI',
                      'emcee': 'MCMC (emcee)',
                      'dynesty': 'MCMC (dynesty)',
                      'fisher': 'Fisher'}

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

# Quijote: https://arxiv.org/pdf/1909.05273, Table 1, top row
# for tau, via raul: "0.0952 is the value we have in the Planck13 cosmo dictionary (in cosmo_parameters.py). It comes from the best fit of Planck+WP+highL+BAO"
cosmo_dict_quijote = {
                'omega_cold'    :  0.3175,
                'omega_baryon'  :  0.049,
                'sigma8_cold'   :  0.834,
                'ns'            :  0.9624,
                'hubble'        :  0.6711,
                'neutrino_mass' :  0.0,
                'w0'            : -1.0,
                'wa'            :  0.0,
                'tau'           :  0.0952,
                }   

cosmo_param_names_ordered = ['omega_cold', 'sigma8_cold', 'hubble', 'omega_baryon', 'ns']
biasparam_names_ordered = ['b1', 'b2', 'bs2', 'bl'] 
param_names_all_ordered = cosmo_param_names_ordered + biasparam_names_ordered

statistics_scaler_funcs = {'pk': 'log_minmax',
                           'bispec': 'minmax',
                          }


def idxs_train_val_test(random_ints, frac_train=0.8, frac_val=0.1, frac_test=0.1,
                        N_tot=None):
    print(frac_train, frac_val, frac_test)
    tol = 1e-6
    assert abs((frac_train+frac_val+frac_test) - 1.0) < tol, "Fractions must add to 1!" 
    if N_tot is None:
        print("Assuming N_tot is the length of random_ints")
        N_tot = len(random_ints)
    int_train = int(frac_train*N_tot)
    int_test = int((1-frac_test)*N_tot)
    print(int_train, int_test)

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


def load_emu(emu_name='lbias_2.0', dir_emus_lbias=None, dir_emus_mpk=None):
    import baccoemu
    if dir_emus_lbias is None:
        dir_emus_lbias = '/cosmos_storage/data_sharing/data_share'
    if dir_emus_mpk is None:
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
    import getdist
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


def get_samples(idx_obs, inf_method, tag_inf, tag_test='', tag_obs=None):
    if inf_method == 'mn':
        return get_samples_mn(idx_obs, tag_inf, tag_test=tag_test)
    if inf_method == 'sbi':
        return get_samples_sbi(idx_obs, tag_inf, tag_test=tag_test)
    elif inf_method == 'emcee':
        return get_samples_emcee(idx_obs, tag_inf, tag_obs=tag_obs)
    elif inf_method == 'dynesty':
        return get_samples_dynesty(idx_obs, tag_inf, tag_obs=tag_obs)
    elif inf_method == 'fisher':
        return get_samples_fisher(idx_obs, tag_inf, tag_test=tag_test)
    else:
        raise ValueError(f'Method {inf_method} not recognized!')
        

def get_moments_test_sbi(tag_inf, tag_test='', param_names=None):
    dir_sbi = f'../results/results_sbi/sbi{tag_inf}'
    fn_samples_test_pred = f'{dir_sbi}/samples_test{tag_test}_pred.npy'
    samples_arr = np.load(fn_samples_test_pred)
    print(samples_arr.shape)

    dir_sbi = f'../results/results_sbi/sbi{tag_inf}'
    param_names_all = np.loadtxt(f'{dir_sbi}/param_names.txt', dtype=str)
    if param_names is None:
        param_names = param_names_all
    i_pn = [list(param_names_all).index(pn) for pn in param_names]
    
    if samples_arr.ndim == 2:
        samples_arr = samples_arr[:,i_pn]
        theta_test_pred = np.mean(samples_arr, axis=0)
        covs_test_pred = np.cov(samples_arr.T)
    elif samples_arr.ndim == 3:
        samples_arr = samples_arr[:,:,i_pn]
        theta_test_pred = np.mean(samples_arr, axis=0)
        covs_test_pred = np.array([np.cov(samples_arr[:,i,:].T) for i in range(samples_arr.shape[1])])
        #theta_test_pred = np.mean(np.mean(samples_arr, axis=0), axis=0)
        #covs_test_pred = np.mean([np.cov(samples_arr[:,i,:].T) for i in range(samples_arr.shape[1])], axis=0)
    else:
        raise ValueError(f"Samples shape {samples_arr.shape} is weird!")
    return theta_test_pred, covs_test_pred, param_names
        
        
def get_moments_test_mn(tag_inf, tag_test=''):
    dir_mn = f'../results/results_moment_network/mn{tag_inf}'
    theta_test_pred = np.load(f'{dir_mn}/theta_test{tag_test}_pred.npy')
    covs_test_pred = np.load(f'{dir_mn}/covs_test{tag_test}_pred.npy')
    return theta_test_pred, covs_test_pred
    
        
def get_samples_mn(idx_obs, tag_inf, tag_test=''):
    rng = np.random.default_rng(42)
    dir_mn = f'../results/results_moment_network/mn{tag_inf}'
    theta_test_pred = np.load(f'{dir_mn}/theta_test{tag_test}_pred.npy')
    covs_test_pred = np.load(f'{dir_mn}/covs_test{tag_test}_pred.npy')
    
    try:
        samples = rng.multivariate_normal(theta_test_pred[idx_obs], 
                                            covs_test_pred[idx_obs], int(1e6),
                                            check_valid='raise')
    except ValueError:
        print("Covariance matrix not PSD! (sampling anyway)")
        samples = rng.multivariate_normal(theta_test_pred[idx_obs], 
                                            covs_test_pred[idx_obs], int(1e6),
                                            check_valid='ignore')
    return samples


def get_samples_sbi(idx_obs, tag_inf, tag_test=''):
    dir_sbi = f'../results/results_sbi/sbi{tag_inf}'
    fn_samples_test_pred = f'{dir_sbi}/samples_test{tag_test}_pred.npy'
    print(f"fn_samples = {fn_samples_test_pred}")
    samples_arr = np.load(fn_samples_test_pred)
    print(samples_arr.shape)
    param_names = np.loadtxt(f'{dir_sbi}/param_names.txt', dtype=str)
    if samples_arr.ndim == 2:
        return samples_arr, param_names
    elif samples_arr.ndim == 3:
        return samples_arr[:,idx_obs,:], param_names
    else:
        raise ValueError(f"Samples shape {samples_arr.shape} is weird!")

def get_samples_emcee(idx_obs, tag_inf, tag_obs=None):
    import emcee
    dir_emcee =  f'../results/results_emcee/samplers{tag_inf}'
    if tag_obs is None:
        tag_obs = f'_idx{idx_obs}'
    fn_emcee = f'{dir_emcee}/sampler{tag_obs}.npy'
    if not os.path.exists(fn_emcee):
        print(f'File {fn_emcee} not found')
        return
    reader = emcee.backends.HDFBackend(fn_emcee)

    tau = reader.get_autocorr_time()
    n_burn = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    #print(n_burn, thin)
    samples = reader.get_chain(discard=n_burn, flat=True, thin=thin)
    
    param_names = np.loadtxt(f'{dir_emcee}/param_names.txt', dtype=str)
    return samples, param_names


def get_samples_dynesty(idx_obs, tag_inf, tag_obs=None):
    dir_dynesty =  f'../results/results_dynesty/samplers{tag_inf}'
    if tag_obs is None:
        tag_obs = f'_idx{idx_obs}'
    fn_dynesty = f'{dir_dynesty}/sampler_results{tag_obs}.npy'
    results_dynesty = np.load(fn_dynesty, allow_pickle=True).item()
    
    # doesn't work upon reload for some reason
    #samples_dynesty = results_dynesty.samples_equal()
    
    from dynesty.utils import resample_equal
    # draw posterior samples
    weights = np.exp(results_dynesty['logwt'] - results_dynesty['logz'][-1])
    samples = resample_equal(results_dynesty.samples, weights)

    param_names = np.loadtxt(f'{dir_dynesty}/param_names.txt', dtype=str)
    return samples, param_names


def get_samples_fisher(idx_obs, tag_inf, tag_test=''):
    dir_fisher = f'../results/results_fisher/fisher{tag_inf}'
    fn_samples_test_pred = f'{dir_fisher}/samples_test{tag_test}_pred.npy'
    print(f"fn_samples = {fn_samples_test_pred}")
    samples_arr = np.load(fn_samples_test_pred)
    print(samples_arr.shape)
    param_names = np.loadtxt(f'{dir_fisher}/param_names.txt', dtype=str)
    if samples_arr.ndim == 2:
        return samples_arr, param_names
    elif samples_arr.ndim == 3:
        return samples_arr[:,idx_obs,:], param_names
    else:
        raise ValueError(f"Samples shape {samples_arr.shape} is weird!")
    

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
            #print(f"Param {pn} not in param dict, adding {sim_name} value")
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


# TODO fill this out with other name mismatches ??
def param_name_to_param_name_emu(param_name):
    # TODO this is not true if have nonzero neutrino mass!! 
    # compute relation: https://chatgpt.com/share/6792b607-1aa8-8002-b6e5-128a98d70302
    if param_name=='sigma8':
        param_name_emu = 'sigma8_cold'
    else:
        param_name_emu = param_name
    return param_name_emu


def get_tracer_field(bias_fields_eul, bias_vector, n_grid_norm=None):
    assert len(bias_vector)==bias_fields_eul.shape[0]-1, "bias_vector must have length one less than number of bias fields"
    if n_grid_norm is None:
        n_grid_norm = bias_fields_eul.shape[-1]
        
    def _sum_bias_fields(fields, bias_vector):
        bias_vector_extended = np.concatenate(([1.0], bias_vector))
        return np.sum([fields[ii]*bias_vector_extended[ii] for ii in range(len(fields))], axis=0)
    
    tracer_field_eul = _sum_bias_fields(bias_fields_eul, bias_vector)
    tracer_field_eul_norm = tracer_field_eul/n_grid_norm**3
    
    return tracer_field_eul_norm



#Compute the predicted galaxy auto pk and galaxy-matter cross pk \
#given a set of bias parameters; combines the pnns

# copied from https://bitbucket.org/rangulo/baccoemu/src/master/baccoemu/lbias_expansion.py
def pnn_to_pk(pnn, bias_params, return_cross=False, pk_type='pk'):
    
    message = 'Please, pass a valid bias array, with' \
            + 'b1, b2, bs2, blaplacian'
    assert len(bias_params) == 4, message

    import itertools
    #k, pnn = self.get_nonlinear_pnn(**kwargs)
    bias_params_extended = np.concatenate(([1], bias_params))
    prod = np.array(
        list(itertools.combinations_with_replacement(np.arange(len(bias_params_extended)),
                                                    r=2)))

    pgal_auto = 0
    for i in range(len(pnn)):
        fac = 2 if prod[i, 0] != prod[i, 1] else 1
        pgal_auto += bias_params_extended[prod[i, 0]] * bias_params_extended[prod[i, 1]] * fac * pnn[i][pk_type]
        
    if return_cross:
        # TODO check this
        pks = [pnn[i]['pk'] for i in range(5)]
        pgal_cross = np.dot(bias_params_extended, pks)
        return pgal_auto, pgal_cross
    else:
        return pgal_auto


# used by scripts/generate_emuPks.py, generate_params_lh()
def generate_randints(n_samples, fn_rands, rng=None, overwrite=False):
    
    if os.path.exists(fn_rands) and not overwrite:
        print(f"Loading from {fn_rands} (already exists)")
        random_ints = np.load(fn_rands, allow_pickle=True)
        return random_ints
    
    if rng is None:
        rng = np.random.default_rng(42)
    
    # save random ints for later train/val/test split
    random_ints = np.arange(n_samples)
    rng.shuffle(random_ints) #in-place
    np.save(fn_rands, random_ints)
    print(f"Saved random ints to {fn_rands}")
    return random_ints


def compute_fisher_matrix(derivatives, covariance_matrix, param_names):
    n_params = len(param_names)
    fisher_matrix = np.zeros((n_params, n_params))
    
    # Invert covariance matrix
    cov_inv = np.linalg.inv(covariance_matrix)
    
    for i, param_i in enumerate(param_names):
        for j, param_j in enumerate(param_names):
            fisher_matrix[i, j] = np.dot(derivatives[param_i], 
                                       np.dot(cov_inv, derivatives[param_j]))
    
    return fisher_matrix