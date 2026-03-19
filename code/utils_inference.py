"""Inference- and sampling-related helpers for muchisimocks.

This file is the canonical home for inference utilities so callers can
import them from a focused module.
"""

import os
import numpy as np

statistics_scaler_funcs = {'pk': 'log_minmax', 'bispec': 'minmax', 'pgm': 'log_minmax_const'}


def get_posterior_maxes(samples_equal, param_names):
    import getdist
    samps = getdist.MCSamples(names=param_names)
    samps.setSamples(samples_equal)
    maxes = []
    for i, pn in enumerate(param_names):
        xvals = np.linspace(min(samples_equal[:, i]), max(samples_equal[:, i]), 1000)
        dens = samps.get1DDensity(pn)
        probs = dens(xvals)
        posterior_max = xvals[np.argmax(probs)]
        maxes.append(posterior_max)
    return maxes


def generate_randints(n_samples, fn_rands, rng=None, overwrite=False):
    if os.path.exists(fn_rands) and not overwrite:
        print(f"Loading from {fn_rands} (already exists)")
        return np.load(fn_rands, allow_pickle=True)
    if rng is None:
        rng = np.random.default_rng(42)
    random_ints = np.arange(n_samples)
    rng.shuffle(random_ints)
    np.save(fn_rands, random_ints)
    print(f"Saved random ints to {fn_rands}")
    return random_ints


def idxs_train_val_test(random_ints, frac_train=0.8, frac_val=0.1, frac_test=0.1,
                        N_tot=None):
    """Split indices into train/val/test from random_ints (e.g. from generate_randints)."""
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
    """Split array into train/val/test using given index arrays."""
    arr_train = arr[idxs_train]
    arr_val = arr[idxs_val]
    arr_test = arr[idxs_test]
    return arr_train, arr_val, arr_test


def get_samples(idx_obs, inf_method, tag_inf, tag_test='', tag_obs=None):
    """Load posterior samples for observation idx_obs (mn, sbi, emcee, dynesty, or fisher)."""
    try:
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
    except FileNotFoundError as e:
        fn = getattr(e, "filename", None)
        fn_part = f": {fn}" if fn else f": {e}"
        print(f"ERROR: missing samples (inf_method={inf_method}, tag_inf={tag_inf}, tag_test={tag_test}){fn_part}")
        return np.array([]), np.array([], dtype=str)


def get_moments_test_sbi(tag_inf, tag_test='', param_names=None):
    """Load SBI test posterior mean and covariances from saved samples."""
    dir_sbi = f'../results/results_sbi/sbi{tag_inf}'
    fn_samples_test_pred = f'{dir_sbi}/samples_test{tag_test}_pred.npy'
    fn_param_names = f'{dir_sbi}/param_names.txt'
    try:
        if not os.path.exists(fn_samples_test_pred):
            raise FileNotFoundError(fn_samples_test_pred)
        if not os.path.exists(fn_param_names):
            raise FileNotFoundError(fn_param_names)

        print(f"fn_samples_test_pred = {fn_samples_test_pred}")
        samples_arr = np.load(fn_samples_test_pred)
        param_names_all = np.loadtxt(fn_param_names, dtype=str)
        if param_names is None:
            param_names = param_names_all
        i_pn = [list(param_names_all).index(pn) for pn in param_names]

        if samples_arr.ndim == 2:
            samples_arr = samples_arr[:, i_pn]
            theta_test_pred = np.mean(samples_arr, axis=0)
            covs_test_pred = np.cov(samples_arr.T)
        elif samples_arr.ndim == 3:
            samples_arr = samples_arr[:, :, i_pn]
            theta_test_pred = np.mean(samples_arr, axis=0)
            covs_test_pred = np.array([np.cov(samples_arr[:, i, :].T) for i in range(samples_arr.shape[1])])
        else:
            raise ValueError(f"Samples shape {samples_arr.shape} is weird!")
        return theta_test_pred, covs_test_pred, param_names
    except FileNotFoundError as e:
        fn = getattr(e, "filename", None)
        fn_part = f": {fn}" if fn else f": {e}"
        print(f"ERROR: missing samples (get_moments_test_sbi, tag_inf={tag_inf}, tag_test={tag_test}){fn_part}")
        return np.array([]), np.array([]), np.array([], dtype=str)


def get_samples_sbi(idx_obs, tag_inf, tag_test=''):
    dir_sbi = f'../results/results_sbi/sbi{tag_inf}'
    fn_samples_test_pred = f'{dir_sbi}/samples_test{tag_test}_pred.npy'
    if not os.path.exists(fn_samples_test_pred):
        raise FileNotFoundError(fn_samples_test_pred)
    print(f"fn_samples = {fn_samples_test_pred}")
    samples_arr = np.load(fn_samples_test_pred)
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
    samples = reader.get_chain(discard=n_burn, flat=True, thin=thin)

    param_names = np.loadtxt(f'{dir_emcee}/param_names.txt', dtype=str)
    return samples, param_names


def get_samples_dynesty(idx_obs, tag_inf, tag_obs=None):
    dir_dynesty =  f'../results/results_dynesty/samplers{tag_inf}'
    if tag_obs is None:
        tag_obs = f'_idx{idx_obs}'
    fn_dynesty = f'{dir_dynesty}/sampler_results{tag_obs}.npy'
    results_dynesty = np.load(fn_dynesty, allow_pickle=True).item()

    from dynesty.utils import resample_equal
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


def reparameterize_theta(theta, param_names):
    """
    Reparameterize theta by multiplying bias and noise parameters by sigma_8.

    For b1, bl, A_b1 (An_b1), and A_bl (An_bl): multiply by sigma_8
    For b2, bs2, A_b2 (An_b2), and A_bs2 (An_bs2): multiply by sigma_8^2

    Parameters:
    -----------
    theta : numpy.ndarray
        Array of shape (n_samples, n_params) or (n_params,) containing parameter values
    param_names : list
        List of parameter names corresponding to columns in theta

    Returns:
    --------
    theta_reparam : numpy.ndarray
        Reparameterized theta array (same shape as input)
    param_names_reparam : list
        List of reparameterized parameter names
    """
    if 'sigma8_cold' not in param_names:
        raise ValueError("sigma8_cold must be in param_names for reparameterization")

    one_d = theta.ndim == 1
    if one_d:
        theta = np.atleast_2d(theta)

    idx_sigma8 = param_names.index('sigma8_cold')
    params_sigma8 = ['b1', 'An_b1', 'bl', 'An_bl']
    params_sigma8_squared = ['b2', 'bs2', 'An_b2', 'An_bs2']

    theta_reparam = theta.copy()
    param_names_reparam = param_names.copy()

    for i, param_name in enumerate(param_names):
        if param_name in params_sigma8:
            sigma8_values = theta[:, idx_sigma8]
            theta_reparam[:, i] = theta[:, i] * sigma8_values
            param_names_reparam[i] = f'sigma8_cold_x_{param_name}'
        elif param_name in params_sigma8_squared:
            sigma8_values = theta[:, idx_sigma8]
            theta_reparam[:, i] = theta[:, i] * (sigma8_values ** 2)
            param_names_reparam[i] = f'sigma8_cold_sq_x_{param_name}'

    if one_d:
        theta_reparam = theta_reparam[0]
    return theta_reparam, param_names_reparam


def reparameterize_bounds(dict_bounds):
    """
    Update parameter bounds when reparameterizing.

    For parameters multiplied by sigma_8, the new bounds are computed as the product
    of the bounds of sigma8_cold and the original parameter.
    For parameters multiplied by sigma_8^2, the new bounds are computed as the product
    of sigma8_cold^2 and the original parameter.
    """
    if 'sigma8_cold' not in dict_bounds:
        raise ValueError("sigma8_cold must be in dict_bounds for reparameterization")

    sigma8_bounds = dict_bounds['sigma8_cold']
    sigma8_low, sigma8_high = sigma8_bounds[0], sigma8_bounds[1]

    params_sigma8 = ['b1', 'An_b1', 'bl', 'An_bl']
    params_sigma8_squared = ['b2', 'bs2', 'An_b2', 'An_bs2']

    dict_bounds_reparam = dict_bounds.copy()

    for param_name in list(dict_bounds.keys()):
        if param_name in params_sigma8:
            param_bounds = dict_bounds[param_name]
            param_low, param_high = param_bounds[0], param_bounds[1]
            products = [sigma8_low * param_low, sigma8_low * param_high,
                        sigma8_high * param_low, sigma8_high * param_high]
            new_low = min(products)
            new_high = max(products)
            new_name = f'sigma8_cold_x_{param_name}'
            dict_bounds_reparam[new_name] = [new_low, new_high]
            del dict_bounds_reparam[param_name]
        elif param_name in params_sigma8_squared:
            param_bounds = dict_bounds[param_name]
            param_low, param_high = param_bounds[0], param_bounds[1]
            products = [sigma8_low**2 * param_low, sigma8_low**2 * param_high,
                        sigma8_high**2 * param_low, sigma8_high**2 * param_high]
            new_low = min(products)
            new_high = max(products)
            new_name = f'sigma8_cold_sq_x_{param_name}'
            dict_bounds_reparam[new_name] = [new_low, new_high]
            del dict_bounds_reparam[param_name]

    return dict_bounds_reparam


def compute_fisher_matrix(derivatives, covariance_matrix, param_names):
    """Compute Fisher matrix F_ij = d_i^T C^{-1} d_j for parameter derivatives."""
    n_params = len(param_names)
    fisher_matrix = np.zeros((n_params, n_params))
    cov_inv = np.linalg.inv(covariance_matrix)
    for i, param_i in enumerate(param_names):
        for j, param_j in enumerate(param_names):
            fisher_matrix[i, j] = np.dot(derivatives[param_i],
                                         np.dot(cov_inv, derivatives[param_j]))
    return fisher_matrix


def chi2(theta_true, theta_pred, covs_pred):
    chi2s = []
    if covs_pred.ndim == 3:
        for t_true, t_pred, cov_pred in zip(theta_true, theta_pred, covs_pred):
            diff = t_true - t_pred
            cov_pred_inv = np.linalg.inv(cov_pred)
            chi2_i = diff.T @ cov_pred_inv @ diff
            chi2s.append(chi2_i.item())
    elif covs_pred.ndim == 2:
        diff = theta_true - theta_pred
        cov_pred_inv = np.linalg.inv(covs_pred)
        chi2_i = diff.T @ cov_pred_inv @ diff
        chi2s.append(chi2_i.item())
    else:
        raise ValueError(f"covs_pred shape {covs_pred.shape} is weird!")
    return chi2s if len(chi2s) > 1 else chi2s[0]


def mse(theta_true, theta_pred):
    return np.mean((theta_true - theta_pred) ** 2, axis=-1)


def figure_of_merit(covs_pred):
    foms = []
    if covs_pred.ndim == 3:
        for cov_pred in covs_pred:
            foms.append(1 / np.sqrt(np.linalg.det(cov_pred)))
    elif covs_pred.ndim == 2:
        foms = 1 / np.sqrt(np.linalg.det(covs_pred))
    else:
        raise ValueError(f"covs_pred shape {covs_pred.shape} is weird!")
    return foms

__all__ = [
    "idxs_train_val_test",
    "split_train_val_test",
    "get_posterior_maxes",
    "get_samples",
    "get_moments_test_sbi",
    "get_samples_sbi",
    "get_samples_emcee",
    "get_samples_dynesty",
    "get_samples_fisher",
    "repeat_arr_rlzs",
    "reparameterize_theta",
    "reparameterize_bounds",
    "compute_fisher_matrix",
    "chi2",
    "mse",
    "figure_of_merit",
    "statistics_scaler_funcs",
    "generate_randints",
]

