"""Auxiliary helpers for moments/sampling and emulator utilities."""

from __future__ import annotations

import numpy as np

import utils_model


def setup_cosmo_emu(cosmo='quijote'):
    print("Setting up emulator cosmology")
    if cosmo == 'quijote':
        cosmo_params = utils_model.cosmo_dict_quijote
    else:
        raise ValueError(f'Cosmo {cosmo} not recognized!')
    return cosmo_params


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
        samples = rng.multivariate_normal(theta_test_pred[idx_obs], covs_test_pred[idx_obs], int(1e6), check_valid='raise')
    except ValueError:
        print("Covariance matrix not PSD! (sampling anyway)")
        samples = rng.multivariate_normal(theta_test_pred[idx_obs], covs_test_pred[idx_obs], int(1e6), check_valid='ignore')
    return samples


def get_cosmo_emu(param_dict, a_scale=1, sim_name='quijote'):
    cosmo = utils_model.get_cosmo(param_dict, a_scale=a_scale, sim_name=sim_name)
    return cosmo_bacco_to_cosmo_baccoemu(cosmo)


def cosmo_bacco_to_cosmo_baccoemu(cosmo):
    param_names_emu = ['sigma8_cold', 'omega_cold', 'hubble', 'ns', 'omega_baryon', 'expfactor', 'neutrino_mass', 'w0', 'wa']
    cosmo_params_emu = {}
    for param_name_emu in param_names_emu:
        if param_name_emu == 'sigma8_cold':
            param_bacco = cosmo.pars['sigma8']
        elif param_name_emu == 'expfactor':
            param_bacco = cosmo.expfactor
        else:
            param_bacco = cosmo.pars[param_name_emu]
        cosmo_params_emu[param_name_emu] = param_bacco
    return cosmo_params_emu


def load_data_emu(statistic, tag_params, tag_biasparams, tag_errG='', tag_datagen='', tag_noiseless='',
                  n_rlzs_per_cosmo=1, tag_mask=''):
    assert statistic == 'pk', "Only implemented for pk for emu"
    tag_mocks = tag_params + tag_biasparams

    assert tag_errG is not None, "tag_errG must be specified"
    dir_emuPk = f'../data/emuPks/emuPks{tag_mocks}'

    assert tag_noiseless in ['', '_noiseless'], "tag_noiseless must be '_noiseless' or ''"
    if 'noiseless' in tag_noiseless:
        assert n_rlzs_per_cosmo == 1, "Why would you want multiple realizations per cosmo if using noiseless?"
        fn_emuPk = f'{dir_emuPk}/emuPks.npy'
    else:
        fn_emuPk = f'{dir_emuPk}/emuPks_noisy{tag_datagen}.npy'
    fn_emuk = f'{dir_emuPk}/emuPks_k.txt'
    fn_emuPkerrG = f'{dir_emuPk}/emuPks_errgaussian{tag_errG}.npy'

    Pk = np.load(fn_emuPk, allow_pickle=True)
    k = np.genfromtxt(fn_emuk)
    print(fn_emuPk)
    print(Pk.shape)

    gaussian_error_pk_orig = np.load(fn_emuPkerrG, allow_pickle=True)
    gaussian_error_pk = np.tile(gaussian_error_pk_orig, (n_rlzs_per_cosmo, 1))
    assert gaussian_error_pk.shape[0] == Pk.shape[0], "Number of pks and errors should be the same, something is wrong"

    idxs_params = np.array([(idx_lh, idx_lh) for idx_lh in range(Pk.shape[0])])
    return k, Pk, gaussian_error_pk, idxs_params


__all__ = [
    "setup_cosmo_emu",
    "get_moments_test_mn",
    "get_samples_mn",
    "get_cosmo_emu",
    "cosmo_bacco_to_cosmo_baccoemu",
    "load_data_emu",
]

