"""Inference- and sampling-related helpers for muchisimocks.

This file is the canonical home for inference utilities so callers can
import them from a focused module.
"""

import os
import numpy as np

import paths

statistics_scaler_funcs = {'pk': 'log_minmax', 'bispec': 'minmax', 'pgm': 'log_minmax_const'}

# --- sigma8 reparameterization (single source of truth; used by inference, bounds, plotting) ---

# Linear in sigma8: value' = value * sigma8_cold
PARAMS_SIGMA8_MULT = ('b1', 'An_b1', 'bl', 'An_bl')
# Quadratic in sigma8: value' = value * sigma8_cold**2
PARAMS_SIGMA8_SQUARED_MULT = ('b2', 'bs2', 'An_b2', 'An_bs2')
REPARAM_PREFIX_X = 'sigma8_cold_x_'
REPARAM_PREFIX_SQ = 'sigma8_cold_sq_x_'


def reparameterized_sample_name(original_pn):
    """Sample/chain column name for `original_pn` after `reparameterize_theta`, or None if not reparameterized."""
    o = str(original_pn)
    if o in PARAMS_SIGMA8_SQUARED_MULT:
        return f'{REPARAM_PREFIX_SQ}{o}'
    if o in PARAMS_SIGMA8_MULT:
        return f'{REPARAM_PREFIX_X}{o}'
    return None


def has_reparameterized_sigma8_columns(param_names):
    """True if any column is a sigma8 product name produced by ``reparameterize_theta``."""
    for n in param_names:
        s = str(n)
        if s.startswith(REPARAM_PREFIX_SQ) and s[len(REPARAM_PREFIX_SQ) :] in PARAMS_SIGMA8_SQUARED_MULT:
            return True
        if s.startswith(REPARAM_PREFIX_X) and s[len(REPARAM_PREFIX_X) :] in PARAMS_SIGMA8_MULT:
            return True
    return False


def forward_reparameterized_value(reparam_column_name, param_names, theta_obs_true):
    """
    Truth value on the *reparameterized* axis (sigma8 * x or sigma8**2 * x) for one column name,
    given ``theta_obs_true`` aligned with physical ``param_names``.

    Returns None if the name is not a known reparameterized column or indices are invalid.
    """
    pn = str(reparam_column_name)
    theta = np.asarray(theta_obs_true, dtype=float).reshape(-1)
    names = list(param_names)

    if pn.startswith(REPARAM_PREFIX_SQ):
        orig = pn[len(REPARAM_PREFIX_SQ) :]
        if orig not in PARAMS_SIGMA8_SQUARED_MULT:
            return None
        power = 2
    elif pn.startswith(REPARAM_PREFIX_X):
        orig = pn[len(REPARAM_PREFIX_X) :]
        if orig not in PARAMS_SIGMA8_MULT:
            return None
        power = 1
    else:
        return None

    if 'sigma8_cold' not in names or orig not in names:
        return None
    i8 = names.index('sigma8_cold')
    io = names.index(orig)
    if i8 >= theta.size or io >= theta.size:
        return None
    return float(theta[io] * (theta[i8] ** power))


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
    dir_sbi = str(paths.DIR_RESULTS / "results_sbi" / f"sbi{tag_inf}")
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
    dir_sbi = str(paths.DIR_RESULTS / "results_sbi" / f"sbi{tag_inf}")
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
    names = list(param_names)
    if 'sigma8_cold' not in names:
        raise ValueError("sigma8_cold must be in param_names for reparameterization")

    one_d = theta.ndim == 1
    if one_d:
        theta = np.atleast_2d(theta)

    idx_sigma8 = names.index('sigma8_cold')
    sigma8_values = theta[:, idx_sigma8]

    theta_reparam = theta.copy()
    param_names_reparam = [str(x) for x in names]

    for i, param_name in enumerate(names):
        if param_name in PARAMS_SIGMA8_MULT:
            theta_reparam[:, i] = theta[:, i] * sigma8_values
            param_names_reparam[i] = f'{REPARAM_PREFIX_X}{param_name}'
        elif param_name in PARAMS_SIGMA8_SQUARED_MULT:
            theta_reparam[:, i] = theta[:, i] * (sigma8_values ** 2)
            param_names_reparam[i] = f'{REPARAM_PREFIX_SQ}{param_name}'

    if one_d:
        theta_reparam = theta_reparam[0]
    return theta_reparam, param_names_reparam


def unreparameterize_theta(theta, param_names, *, strict=True):
    """
    Inverse of ``reparameterize_theta``: recover physical bias/noise columns from
    ``sigma8_cold_x_*`` / ``sigma8_cold_sq_x_*`` using each row's ``sigma8_cold``.

    ``theta`` may be ``(n_params,)``, ``(n_samples, n_params)``, or
    ``(n_batch, n_samples, n_params)`` (same ``param_names`` for every row).

    If there are sigma8-product columns but ``sigma8_cold`` is missing: when
    ``strict`` is True (default), raises ``ValueError``; when False, prints a
    warning and returns the input unchanged (same shape, same names).
    """
    theta = np.asarray(theta, dtype=float)
    original_ndim = theta.ndim
    if original_ndim not in (1, 2, 3):
        raise ValueError(
            f"unreparameterize_theta expects theta.ndim in (1, 2, 3), got {original_ndim}"
        )

    one_d = original_ndim == 1
    restore_batch = original_ndim == 3
    if one_d:
        theta_work = np.atleast_2d(theta)
    elif restore_batch:
        b, n, p = theta.shape
        theta_work = theta.reshape(-1, p)
    else:
        theta_work = theta

    param_names_list = [str(n) for n in param_names]

    needs_transform = has_reparameterized_sigma8_columns(param_names_list)
    if not needs_transform:
        return theta.copy(), np.asarray(param_names_list, dtype=str)

    if 'sigma8_cold' not in param_names_list:
        if strict:
            raise ValueError(
                "sigma8_cold must be in param_names to unreparameterize sigma8 product columns"
            )
        print(
            "Warning: Need sigma8_cold for unreparameterization but not found in chain. "
            "Skipping unreparameterization for this chain."
        )
        return theta.copy(), np.asarray(param_names_list, dtype=str)

    idx_sigma8 = param_names_list.index('sigma8_cold')
    sigma8_values = theta_work[:, idx_sigma8]

    theta_out = theta_work.copy()
    param_names_out = []

    for i, name in enumerate(param_names_list):
        if name.startswith(REPARAM_PREFIX_SQ):
            base = name[len(REPARAM_PREFIX_SQ) :]
            if base in PARAMS_SIGMA8_SQUARED_MULT:
                theta_out[:, i] = theta_work[:, i] / (sigma8_values ** 2)
                param_names_out.append(base)
                continue
        elif name.startswith(REPARAM_PREFIX_X):
            base = name[len(REPARAM_PREFIX_X) :]
            if base in PARAMS_SIGMA8_MULT:
                theta_out[:, i] = theta_work[:, i] / sigma8_values
                param_names_out.append(base)
                continue
        theta_out[:, i] = theta_work[:, i]
        param_names_out.append(name)

    names_arr = np.array(param_names_out, dtype=str)

    if restore_batch:
        theta_out = theta_out.reshape(b, n, -1)
    elif one_d:
        theta_out = theta_out[0]
    return theta_out, names_arr


def unreparameterize_prediction_block(theta_pred, theta_true, param_names, covs_per_sample=None):
    """
    Unreparameterize one prediction block: ``theta_pred`` / ``theta_true`` shaped
    ``(n_samples, n_params)``, optional per-sample covariances ``(n_samples, n_params, n_params)``
    in reparameterized column order. Uses predicted ``sigma8_cold`` per row for the
    covariance scaling (pre-unreparameterization values).
    """
    names = list(param_names)
    idx_sigma8 = names.index('sigma8_cold')
    tp = np.asarray(theta_pred, dtype=float)
    sigma8_vals = tp[:, idx_sigma8]
    theta_pred_u, names_out = unreparameterize_theta(theta_pred, names)
    theta_true_u, _ = unreparameterize_theta(theta_true, names)
    cov_out = None
    if covs_per_sample is not None:
        c = np.asarray(covs_per_sample, dtype=float)
        cov_out = np.empty_like(c)
        for j in range(c.shape[0]):
            cov_out[j] = scale_covariance_unreparameterize_approx(
                c[j], names, float(sigma8_vals[j])
            )
    return theta_pred_u, theta_true_u, cov_out, list(names_out)


def scale_covariance_unreparameterize_approx(cov, param_names, sigma8_value):
    """
    Approximate covariance map from reparameterized to physical space (linearized inverse;
    rows/cols involving ``sigma8_cold`` with a reparameterized parameter are zeroed).

    Parameters
    ----------
    cov : (n, n) array
    param_names : sequence of str
        Names in the same order as rows/columns of ``cov`` (reparameterized space).
    sigma8_value : float
        sigma8_cold value for this sample (e.g. from predicted mean).

    Returns
    -------
    cov_scaled : ndarray
        Copy of ``cov`` after scaling.
    """
    param_names_list = [str(p) for p in param_names]
    cov = np.asarray(cov, dtype=float).copy()
    idx_sigma8 = param_names_list.index('sigma8_cold')

    reparam_to_orig = {}
    for i, pn in enumerate(param_names_list):
        if pn.startswith(REPARAM_PREFIX_SQ):
            orig = pn[len(REPARAM_PREFIX_SQ) :]
            if orig in PARAMS_SIGMA8_SQUARED_MULT:
                reparam_to_orig[i] = orig
        elif pn.startswith(REPARAM_PREFIX_X):
            orig = pn[len(REPARAM_PREFIX_X) :]
            if orig in PARAMS_SIGMA8_MULT:
                reparam_to_orig[i] = orig

    for reparam_idx, orig_param_name in reparam_to_orig.items():
        if orig_param_name in PARAMS_SIGMA8_MULT:
            cov[reparam_idx, reparam_idx] = cov[reparam_idx, reparam_idx] / (sigma8_value ** 2)
            for k in range(len(param_names_list)):
                if k != reparam_idx:
                    if k == idx_sigma8:
                        cov[reparam_idx, k] = 0.0
                        cov[k, reparam_idx] = 0.0
                    else:
                        cov[reparam_idx, k] = cov[reparam_idx, k] / sigma8_value
                        cov[k, reparam_idx] = cov[k, reparam_idx] / sigma8_value
        elif orig_param_name in PARAMS_SIGMA8_SQUARED_MULT:
            cov[reparam_idx, reparam_idx] = cov[reparam_idx, reparam_idx] / (sigma8_value ** 4)
            for k in range(len(param_names_list)):
                if k != reparam_idx:
                    if k == idx_sigma8:
                        cov[reparam_idx, k] = 0.0
                        cov[k, reparam_idx] = 0.0
                    else:
                        cov[reparam_idx, k] = cov[reparam_idx, k] / (sigma8_value ** 2)
                        cov[k, reparam_idx] = cov[k, reparam_idx] / (sigma8_value ** 2)

    return cov


def get_chain_statistics(
    idx_obs,
    inf_method,
    tag_inf,
    tag_test='',
    tag_obs=None,
    unreparameterize=False,
):
    """Marginalized median and 16/84 percentiles per parameter.

    If ``unreparameterize``, apply ``unreparameterize_theta`` first.
    """
    samples, param_names = get_samples(
        idx_obs, inf_method, tag_inf, tag_test=tag_test, tag_obs=tag_obs
    )
    if samples.size == 0 or len(param_names) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([], dtype=str)
    if samples.ndim != 2:
        raise ValueError(f"Samples shape {samples.shape} is weird!")

    if unreparameterize:
        samples, param_names = unreparameterize_theta(samples, param_names)

    pct16, median, pct84 = np.percentile(samples, [16, 50, 84], axis=0)
    return median, pct16, pct84, param_names


def reparameterize_bounds(dict_bounds):
    """
    Update parameter bounds when reparameterizing.

    Uses ``reparameterize_theta`` as the single source of truth: for each parameter
    that is multiplied by ``sigma8_cold`` or ``sigma8_cold**2``, the new interval is
    the min/max of that map over the four corners of the ``(sigma8_cold, param)``
    rectangle (other coordinates are fixed to their lower bounds and do not enter
    the monomial). Unchanged parameters keep ``[lo, hi]`` from the original dict.
    """
    if 'sigma8_cold' not in dict_bounds:
        raise ValueError("sigma8_cold must be in dict_bounds for reparameterization")

    names = list(dict_bounds.keys())
    lo_vec = np.array([dict_bounds[k][0] for k in names], dtype=float)
    hi_vec = np.array([dict_bounds[k][1] for k in names], dtype=float)
    idx_s8 = names.index('sigma8_cold')

    # Column order / reparameterized names match ``reparameterize_theta``
    _, names_reparam = reparameterize_theta(lo_vec, names)
    names_reparam = list(names_reparam)

    bounds_map = {}

    for i, pn in enumerate(names):
        if pn in PARAMS_SIGMA8_MULT or pn in PARAMS_SIGMA8_SQUARED_MULT:
            rows = []
            for s8 in (lo_vec[idx_s8], hi_vec[idx_s8]):
                for pv in (lo_vec[i], hi_vec[i]):
                    corner = lo_vec.copy()
                    corner[idx_s8] = s8
                    corner[i] = pv
                    rows.append(corner)
            theta_r, nr = reparameterize_theta(np.stack(rows, axis=0), names)
            new_name = str(nr[i])
            bounds_map[new_name] = [
                float(theta_r[:, i].min()),
                float(theta_r[:, i].max()),
            ]
        else:
            bounds_map[pn] = [float(lo_vec[i]), float(hi_vec[i])]

    return {pn: bounds_map[pn] for pn in names_reparam}


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
    "get_chain_statistics",
    "get_moments_test_sbi",
    "get_samples_sbi",
    "get_samples_emcee",
    "get_samples_dynesty",
    "get_samples_fisher",
    "repeat_arr_rlzs",
    "PARAMS_SIGMA8_MULT",
    "PARAMS_SIGMA8_SQUARED_MULT",
    "REPARAM_PREFIX_X",
    "REPARAM_PREFIX_SQ",
    "reparameterized_sample_name",
    "has_reparameterized_sigma8_columns",
    "unreparameterize_prediction_block",
    "forward_reparameterized_value",
    "scale_covariance_unreparameterize_approx",
    "reparameterize_theta",
    "unreparameterize_theta",
    "reparameterize_bounds",
    "compute_fisher_matrix",
    "chi2",
    "mse",
    "figure_of_merit",
    "statistics_scaler_funcs",
    "generate_randints",
]

