import os
os.environ["OMP_NUM_THREADS"] = str(1)

import multiprocessing as mp
import numpy as np
import pathlib
import time



### Emcee
def log_prior(theta):
    for pp in range(len(_param_names)):
       if (theta[pp] < _dict_bounds[_param_names[pp]][0]) or (theta[pp] >= _dict_bounds[_param_names[pp]][1]):
           return -np.inf
    return 0.0


def log_likelihood(theta):
    for pp in range(len(_param_names)):
        _cosmo_params[_emu_param_names[pp]] = theta[pp]
    expfactor = 1.0 # careful, may need to change at some point!
    _cosmo_params['expfactor'] = expfactor
    _, pk_model_unscaled, _ = _emu.get_galaxy_real_pk(bias=_bias_params, k=_k, 
                                                **_cosmo_params)
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
    for pp in range(len(_param_names)):
        width = _dict_bounds[_param_names[pp]][1] - _dict_bounds[_param_names[pp]][0]
        min_bound = _dict_bounds[_param_names[pp]][0]
        
        u_t = width*u[pp] + min_bound
        u_transformed.append(u_t)           

    return np.array(u_transformed)


def evaluate_dynesty(idx_test, pk_data, cov_inv, scaler,
                     emu, cosmo_params, bias_params, k,
                     dict_bounds, param_names, emu_param_names,
                     tag_inf='', n_threads=10):
    
    # import here so if only have emcee, that still works
    import dynesty

    dir_dynesty =  f'../results/results_dynesty/samplers{tag_inf}'
    p = pathlib.Path(dir_dynesty)
    p.mkdir(parents=True, exist_ok=True)
    fn_dynesty = f'{dir_dynesty}/sampler_results_idxtest{idx_test}.npy'
    if os.path.exists(fn_dynesty):
        print(f"Sampler results file {fn_dynesty} already exists, skipping")
        return
    
    assert len(emu_param_names) == len(param_names), "Parameter names and emulator parameter names must be the same length"
    global _pk_data, _cov_inv, _scaler
    global _emu, _cosmo_params, _bias_params, _k
    global _dict_bounds, _param_names, _emu_param_names
    
    # i think i should maybe only pass param names, and then in here 
    # if they don't align with emu_param_names, do the translation??
    _pk_data, _cov_inv, _scaler = pk_data, cov_inv, scaler
    _emu, _cosmo_params, _bias_params, _k =  emu, cosmo_params, bias_params, k
    _dict_bounds, _param_names, _emu_param_names = dict_bounds, param_names, emu_param_names
    
    start = time.time()
    n_params = len(param_names)
    with dynesty.pool.Pool(n_threads, log_likelihood, prior_transform) as pool:
        sampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform, n_params, 
                                             nlive=50, bound='single')
        sampler.run_nested(dlogz=0.1)
    end = time.time()
    print(f"Time: {end-start} s ({(end-start)/60} min)")
    
    results_dynesty = sampler.results
    #samples_dynesty = results_dynesty.samples_equal()
    #print(samples_dynesty.shape)
    
    np.save(fn_dynesty, results_dynesty)


def evaluate_emcee(idx_test, pk_data, cov_inv, scaler,
                     emu, cosmo_params, bias_params, k,
                     dict_bounds, param_names, emu_param_names,
                     tag_inf='', n_threads=8):
    
    # import here so if only want emcee (not dynesty), still runs
    import emcee

    global _pk_data, _cov_inv, _scaler
    global _emu, _cosmo_params, _bias_params, _k
    global _dict_bounds, _param_names, _emu_param_names
    
    _pk_data, _cov_inv, _scaler = pk_data, cov_inv, scaler
    _emu, _cosmo_params, _bias_params, _k =  emu, cosmo_params, bias_params, k
    _dict_bounds, _param_names, _emu_param_names = dict_bounds, param_names, emu_param_names
    
    n_params = len(param_names)
    
    # n_burn = 100
    n_steps = 4000 # 50000
    n_walkers = 4 * n_params

    dir_emcee =  f'../results/results_emcee/samplers{tag_inf}'
    p = pathlib.Path(dir_emcee)
    p.mkdir(parents=True, exist_ok=True)
    fn_emcee = f'{dir_emcee}/sampler_idxtest{idx_test}.npy'
    if os.path.exists(fn_emcee):
        print(f"Sampler results file {fn_emcee} already exists, skipping")
        return
    
    backend = emcee.backends.HDFBackend(fn_emcee)
    backend.reset(n_walkers, n_params)

    rng = np.random.default_rng(seed=42)
    theta_0 = np.array([[rng.uniform(low=dict_bounds[param_name][0],high=dict_bounds[param_name][1]) 
                        for param_name in param_names] for _ in range(n_walkers)])

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

    print(f"Time: {end-start} s ({(end-start)/60} min)")

    # saving done thru emcee backend! generate samples later, directly from saved sampler
    #samples_emcee = sampler_emcee.get_chain(discard=n_burn, flat=True, thin=1)
    #np.save(samples_emcee, fn_emcee)
