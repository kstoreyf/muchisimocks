import os
os.environ["OMP_NUM_THREADS"] = str(1)

import multiprocessing as mp
import numpy as np
import pathlib
import time

import utils



### Emcee
def log_prior(theta):
    for pp in range(len(_param_names_vary)):
       if (theta[pp] < _dict_bounds[_param_names_vary[pp]][0]) or (theta[pp] >= _dict_bounds[_param_names_vary[pp]][1]):
           return -np.inf
    return 0.0


def log_likelihood(theta):
    # theta combines cosmo and bias params sampling over, and names are _param_names
    #for pp in range(len(_cosmo_param_names)):
    cosmo_params = _cosmo_param_dict_fixed.copy()
    for cosmo_param_name in _cosmo_param_names_vary:
        i_param = _param_names_vary.index(cosmo_param_name)
        cosmo_param_name_emu = utils.param_name_to_param_name_emu(cosmo_param_name)
        cosmo_params[cosmo_param_name_emu] = theta[i_param]
                
    expfactor = 1.0 # careful, may need to change at some point!
    cosmo_params['expfactor'] = expfactor
    
    bias_params = _bias_param_dict_fixed.copy()
    for bias_param_name in _bias_param_names_vary:
        i_param = _param_names_vary.index(bias_param_name)
        bias_params[bias_param_name] = theta[i_param]
    bias_vector = [bias_params[bname] for bname in _bias_param_names_ordered]

    _, pk_model_unscaled, _ = _emu.get_galaxy_real_pk(bias=bias_vector, k=_k, 
                                                **cosmo_params)
    pk_model = _scaler.scale(pk_model_unscaled)
    diff = _pk_data-pk_model

    return -0.5*np.dot(diff,np.dot(_cov_inv,diff))

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


### Dynesty
def prior_transform(u):

    u_transformed = []
    for pp in range(len(_param_names_vary)):
        width = _dict_bounds[_param_names_vary[pp]][1] - _dict_bounds[_param_names_vary[pp]][0]
        min_bound = _dict_bounds[_param_names_vary[pp]][0]
        
        u_t = width*u[pp] + min_bound
        u_transformed.append(u_t)           

    return np.array(u_transformed)


def evaluate_mcmc(idx_obs, pk_data, cov_inv, scaler, 
                   emu, k, 
                   cosmo_param_dict_fixed, bias_param_dict_fixed, 
                   cosmo_param_names_vary, bias_param_names_vary,
                   dict_bounds_cosmo, dict_bounds_bias,
                   tag_inf='', tag_obs=None, n_threads=8, mcmc_framework='dynesty'):

    global _pk_data, _cov_inv, _scaler
    global _emu, _k
    global _cosmo_param_dict_fixed, _bias_param_dict_fixed
    global _cosmo_param_names_vary, _bias_param_names_vary, _param_names_vary
    global _dict_bounds
    global _bias_param_names_ordered

    # for some reason using "update" does not work, at least if one dict is empty
    dict_bounds = {**dict_bounds_cosmo, **dict_bounds_bias}
    _bias_param_names_ordered = ['b1', 'b2', 'bs2', 'bl']    
    _pk_data, _cov_inv, _scaler = pk_data, cov_inv, scaler
    _emu, _k, _dict_bounds, = emu, k, dict_bounds
    _cosmo_param_dict_fixed, _bias_param_dict_fixed = cosmo_param_dict_fixed, bias_param_dict_fixed
    _cosmo_param_names_vary, _bias_param_names_vary = cosmo_param_names_vary, bias_param_names_vary    
    _param_names_vary = _cosmo_param_names_vary + _bias_param_names_vary

    if mcmc_framework == 'dynesty':
        evaluate_dynesty(idx_obs, tag_inf=tag_inf, tag_obs=tag_obs, n_threads=n_threads)
    elif mcmc_framework == 'emcee':
        evaluate_emcee(idx_obs, tag_inf=tag_inf, tag_obs=tag_obs, n_threads=n_threads)
    
    
def evaluate_dynesty(idx_obs, tag_inf='', tag_obs=None, n_threads=8):
    
    # import here so if only have emcee, that still works
    import dynesty

    dir_dynesty =  f'../results/results_dynesty/samplers{tag_inf}'
    p = pathlib.Path(dir_dynesty)
    p.mkdir(parents=True, exist_ok=True)
    if tag_obs is None:
        tag_obs = f'_idx{idx_obs}'
    fn_dynesty = f'{dir_dynesty}/sampler_results{tag_obs}.npy'
    if os.path.exists(fn_dynesty):
        print(f"Sampler results file {fn_dynesty} already exists, skipping")
        return
    
    start = time.time()
    n_params = len(_cosmo_param_names_vary) + len(_bias_param_names_vary)
    with dynesty.pool.Pool(n_threads, log_likelihood, prior_transform) as pool:
        sampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform, n_params, 
                                             nlive=50, bound='single')
        sampler.run_nested(dlogz=0.1)
    end = time.time()
    print(f"Time: {end-start} s ({(end-start)/60} min)")
    
    results_dynesty = sampler.results
    #samples_dynesty = results_dynesty.samples_equal()
    #print(samples_dynesty.shape)
    
    print(f"Saving parameters to {_param_names_vary}")
    np.save(fn_dynesty, results_dynesty)

    print(f"Saving parameters to {_param_names_vary}")
    np.savetxt(f'{dir_dynesty}/param_names.txt', _param_names_vary, fmt='%s')
    
    
def evaluate_emcee(idx_obs, tag_inf='', tag_obs=None, n_threads=8):
    
    # import here so if only want emcee (not dynesty), still runs
    import emcee

    n_params = len(_cosmo_param_names_vary) + len(_bias_param_names_vary)

    # n_burn = 100
    n_steps = 4000 # 50000
    n_walkers = 4 * n_params

    dir_emcee =  f'../results/results_emcee/samplers{tag_inf}'
    p = pathlib.Path(dir_emcee)
    p.mkdir(parents=True, exist_ok=True)
    if tag_obs is None:
        tag_obs = f'_idx{idx_obs}'
    fn_emcee = f'{dir_emcee}/sampler{tag_obs}.npy'
    if os.path.exists(fn_emcee):
        print(f"Sampler results file {fn_emcee} already exists, skipping")
        return
    
    backend = emcee.backends.HDFBackend(fn_emcee)
    backend.reset(n_walkers, n_params)

    rng = np.random.default_rng(seed=42)
    theta_0 = np.array([[rng.uniform(low=_dict_bounds[param_name][0],high=_dict_bounds[param_name][1]) 
                        for param_name in _param_names_vary] for _ in range(n_walkers)])

    start = time.time()
    if n_threads>1:
        with mp.Pool(processes=n_threads) as pool:
            sampler_emcee = emcee.EnsembleSampler(n_walkers, n_params, log_posterior, 
                                                  pool=pool, backend=backend)
            _ = sampler_emcee.run_mcmc(theta_0, n_steps, progress=True) 
    else:
        sampler_emcee = emcee.EnsembleSampler(n_walkers, n_params, log_posterior,
                                              backend=backend)
        _ = sampler_emcee.run_mcmc(theta_0, n_steps, progress=True) 
    end = time.time()

    print(f"Saving parameters to {_param_names_vary}")
    np.savetxt(f'{dir_emcee}/param_names.txt', _param_names_vary, fmt='%s')

    print(f"Time: {end-start} s ({(end-start)/60} min)")

    # saving done thru emcee backend! generate samples later, directly from saved sampler
    #samples_emcee = sampler_emcee.get_chain(discard=n_burn, flat=True, thin=1)
    #np.save(samples_emcee, fn_emcee)
