"""Functions moved out of code/utils_inference.py in the v2 cleanup (2026-11-15).

No longer used by the pipeline; kept for reference / possible reuse, not maintained.
Imports are copied from code/utils_inference.py; module-level names come from the star import.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import os
import numpy as np
import paths
from utils_inference import *  # noqa: F401,F403


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


def mse(theta_true, theta_pred):
    return np.mean((theta_true - theta_pred) ** 2, axis=-1)


