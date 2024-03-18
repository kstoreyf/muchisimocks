import os
os.environ["OMP_NUM_THREADS"] = str(1)

import emcee
import dynesty
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


def evaluate_dynesty(pk_data, cov_inv, scaler,
                     emu, cosmo_params, bias_params, k,
                     dict_bounds, param_names, emu_param_names,
                     n_threads=8):
    
    global _pk_data, _cov_inv, _scaler
    global _emu, _cosmo_params, _bias_params, _k
    global _dict_bounds, _param_names, _emu_param_names
    
    _pk_data, _cov_inv, _scaler = pk_data, cov_inv, scaler
    _emu, _cosmo_params, _bias_params, _k =  emu, cosmo_params, bias_params, k
    _dict_bounds, _param_names, _emu_param_names = dict_bounds, param_names, emu_param_names
    
    n_params = len(param_names)
    with dynesty.pool.Pool(n_threads, log_likelihood, prior_transform) as pool:
        sampler_test = dynesty.NestedSampler(pool.loglike, pool.prior_transform, n_params, 
                                            nlive=20, bound='single')
        sampler_test.run_nested(dlogz=0.01)
        
    results_test = sampler_test.results
    samples_dynesty_test = results_test.samples_equal()
    print(samples_dynesty_test.shape)


def evaluate_emcee(idx_test, pk_data, cov_inv, scaler,
                     emu, cosmo_params, bias_params, k,
                     dict_bounds, param_names, emu_param_names,
                     tag_inf='', n_threads=8):
    
    global _pk_data, _cov_inv, _scaler
    global _emu, _cosmo_params, _bias_params, _k
    global _dict_bounds, _param_names, _emu_param_names
    
    _pk_data, _cov_inv, _scaler = pk_data, cov_inv, scaler
    _emu, _cosmo_params, _bias_params, _k =  emu, cosmo_params, bias_params, k
    _dict_bounds, _param_names, _emu_param_names = dict_bounds, param_names, emu_param_names
    
    n_params = len(param_names)
    
    # n_burn = 100
    n_steps = 5000 # 50000
    n_walkers = 4 * n_params

    dir_emcee =  f'../data/results_emcee/samplers{tag_inf}'
    p = pathlib.Path(dir_emcee)
    p.mkdir(parents=True, exist_ok=True)
    fn_emcee = f'{dir_emcee}/sampler_idxtest{idx_test}.npy'
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

    #samples_emcee = sampler_emcee.get_chain(discard=n_burn, flat=True, thin=1)
    #np.save(samples_emcee, fn_emcee)
