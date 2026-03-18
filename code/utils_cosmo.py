"""Cosmology- and emulator-related helpers for muchisimocks."""

import numpy as np

from .utils import (
    param_label_dict,
    cosmo_dict_quijote,
    cosmo_dict_shame,
    bias_dict_shame,
    cosmo_param_names_ordered,
    biasparam_names_ordered,
    param_names_all_ordered,
    noiseparam_names_ordered,
)


def setup_cosmo_emu(cosmo: str = "quijote"):
    """Return cosmology parameter dict for a known emulator cosmology."""
    print("Setting up emulator cosmology")
    if cosmo == "quijote":
        cosmo_params = cosmo_dict_quijote
    else:
        raise ValueError(f"Cosmo {cosmo} not recognized!")
    return cosmo_params


def param_dict_to_bacco_param_dict(param_dict, neutrino_mass=None):
    """Convert our param names/units to bacco Cosmology format (e.g. sigma8_cold -> sigma8)."""
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

    if 'omega_cdm' in param_dict_bacco:
        pass
    elif 'omega_cold' in param_dict_bacco:
        param_dict_bacco['omega_cdm'] = param_dict_bacco['omega_cold'] - param_dict_bacco['omega_baryon']
    elif 'omega_m' in param_dict_bacco:
        param_dict_bacco['omega_cold'] = param_dict_bacco['omega_m'] - neutrino_mass
        param_dict_bacco['omega_cdm'] = param_dict_bacco['omega_cold'] - param_dict_bacco['omega_baryon']

    return param_dict_bacco


def get_cosmo(param_dict, a_scale=1, sim_name='quijote'):
    """Build bacco Cosmology from param dict; fills missing keys from fiducial (e.g. quijote)."""
    import bacco

    param_names_bacco = ['omega_cdm', 'omega_baryon', 'hubble', 'ns', 'sigma8',
                         'tau', 'neutrino_mass', 'w0', 'wa']

    if sim_name == 'quijote':
        param_dict_bacco_fid = param_dict_to_bacco_param_dict(cosmo_dict_quijote)
    else:
        raise ValueError(f'Simulation {sim_name} not recognized!')

    neutrino_mass = param_dict['neutrino_mass'] if 'neutrino_mass' in param_dict \
        else param_dict_bacco_fid['neutrino_mass']

    param_dict_bacco = param_dict_to_bacco_param_dict(param_dict, neutrino_mass=neutrino_mass)

    cosmopars = {}
    for pn in param_names_bacco:
        if pn in param_dict_bacco:
            cosmopars[pn] = param_dict_bacco[pn]
        else:
            cosmopars[pn] = param_dict_bacco_fid[pn]

    cosmo = bacco.Cosmology(**cosmopars, verbose=False)
    cosmo.set_expfactor(a_scale)
    return cosmo


def get_cosmo_emu(param_dict, a_scale=1, sim_name='quijote'):
    cosmo = get_cosmo(param_dict, a_scale=a_scale, sim_name=sim_name)
    return cosmo_bacco_to_cosmo_baccoemu(cosmo)


def cosmo_bacco_to_cosmo_baccoemu(cosmo):
    """Convert a bacco Cosmology object into the parameter dict expected by baccoemu."""
    param_names_emu = ['sigma8_cold', 'omega_cold', 'hubble', 'ns', 'omega_baryon',
                       'expfactor', 'neutrino_mass', 'w0', 'wa']
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


# TODO fill this out with other name mismatches ??
def param_name_to_param_name_emu(param_name):
    """Map bacco param name to emulator name (e.g. sigma8 -> sigma8_cold). Not exact for nonzero neutrino_mass."""
    # TODO this is not true if have nonzero neutrino mass!!
    # compute relation: https://chatgpt.com/share/6792b607-1aa8-8002-b6e5-128a98d70302
    if param_name == 'sigma8':
        param_name_emu = 'sigma8_cold'
    else:
        param_name_emu = param_name
    return param_name_emu


def get_tracer_field(bias_fields_eul, bias_vector, n_grid_norm,
                     noise_field=None, A_noise=None, noise_model='multiplicative'):
    assert len(bias_vector) == bias_fields_eul.shape[0] - 1, "bias_vector must have length one less than number of bias fields"

    def _sum_bias_fields(fields, bias_vector):
        bias_vector_extended = np.concatenate(([1.0], bias_vector))
        return np.sum([fields[ii] * bias_vector_extended[ii] for ii in range(len(fields))], axis=0)

    tracer_field_eul = _sum_bias_fields(bias_fields_eul, bias_vector)
    tracer_field_eul_norm = tracer_field_eul / n_grid_norm**3

    if noise_field is not None:
        assert A_noise is not None, "Must provide A_noise if noise_field is provided"
        if noise_field.shape != tracer_field_eul.shape:
            raise ValueError(f"Noise field shape {noise_field.shape} does not match tracer field shape {tracer_field_eul.shape}")
        if noise_model == 'multiplicative':
            assert len(A_noise) == len(bias_fields_eul), "A_noise must have same length as bias fields (5)"
            tracer_field_noise = np.sum([bias_fields_eul[ii] * A_noise[ii] * noise_field
                                         for ii in range(len(bias_fields_eul))], axis=0)
            tracer_field_noise /= n_grid_norm**3
        elif noise_model == 'additive':
            assert isinstance(A_noise, (float, int)), "A_noise must be a single number for additive noise"
            tracer_field_noise = A_noise * noise_field
        else:
            raise ValueError(f"Noise type {noise_model} not recognized!")
        tracer_field_eul_norm += tracer_field_noise

    return tracer_field_eul_norm


def pnn_to_pk(pnn, bias_params, return_cross=False, pk_type='pk'):
    """Compute galaxy auto power spectrum from PNN and bias parameters."""
    message = 'Please, pass a valid bias array, with' \
              + 'b1, b2, bs2, blaplacian'
    assert len(bias_params) == 4, message

    import itertools
    bias_params_extended = np.concatenate(([1], bias_params))
    prod = np.array(
        list(itertools.combinations_with_replacement(np.arange(len(bias_params_extended)),
                                                     r=2)))

    pgal_auto = 0
    for i in range(len(pnn)):
        fac = 2 if prod[i, 0] != prod[i, 1] else 1
        pgal_auto += bias_params_extended[prod[i, 0]] * bias_params_extended[prod[i, 1]] * fac * pnn[i][pk_type]

    return pgal_auto


def pnn_to_pgm(pnn, bias_params, pk_type='pk'):
    """
    Compute galaxy-matter cross power P_gm from PNN, matching compute_pgm() which uses matter = bias_terms_eul[1].
    P_gm = <tracer, field_1> = P_01 + b1*P_11 + b2*P_21 + bs2*P_31 + bl*P_41.
    PNN indices for (i,1) with i=0..4: 1, 5, 6, 7, 8.
    """
    message = 'Please, pass a valid bias array, with b1, b2, bs2, bl'
    assert len(bias_params) == 4, message
    bias_params_extended = np.concatenate(([1], bias_params))
    pnn_idx_pgm = [1, 5, 6, 7, 8]
    pks_cross = [pnn[i][pk_type] for i in pnn_idx_pgm]
    return np.dot(bias_params_extended, pks_cross)


def remove_highk_modes(field, box_size_mock, n_grid_target):
    """
    Remove high-k modes from a field to downsample it to a target grid size.
    """
    import bacco
    import pyfftw

    n_grid = field.shape[-1]
    k_nyq = np.pi / box_size_mock * n_grid_target
    kmesh = bacco.visualization.np_get_kmesh((n_grid, n_grid, n_grid), box_size_mock, real=True)
    mask = (kmesh[:,:,:,0]<=k_nyq) & (kmesh[:,:,:,1]<=k_nyq) & (kmesh[:,:,:,2]<=k_nyq) & \
           (kmesh[:,:,:,0]>-k_nyq) & (kmesh[:,:,:,1]>-k_nyq) & (kmesh[:,:,:,2]>-k_nyq)
    assert n_grid_target % 2 == 0, "n_grid_target must be even!"

    deltak = pyfftw.builders.rfftn(field, auto_align_input=False, auto_contiguous=False, avoid_copy=True)
    deltakcut = deltak()[mask]
    deltakcut = deltakcut.reshape(n_grid_target, n_grid_target, int(n_grid_target/2)+1)
    field_kcut = pyfftw.builders.irfftn(deltakcut, axes=(0,1,2))()
    return field_kcut


def remove_highk_modes_velocity(velocity_field, box_size, n_grid_target):
    """
    Downsample velocity field by removing high-k modes.
    """
    import bacco
    import pyfftw

    n_grid = velocity_field.shape[-1]
    k_nyq = np.pi / box_size * n_grid_target
    kmesh = bacco.visualization.np_get_kmesh((n_grid, n_grid, n_grid), box_size, real=True)
    mask = (kmesh[:,:,:,0]<=k_nyq) & (kmesh[:,:,:,1]<=k_nyq) & (kmesh[:,:,:,2]<=k_nyq) & \
           (kmesh[:,:,:,0]>-k_nyq) & (kmesh[:,:,:,1]>-k_nyq) & (kmesh[:,:,:,2]>-k_nyq)

    velocity_field_kcut = []
    assert n_grid_target % 2 == 0, "n_grid_target must be even!"

    for component_id in range(3):  # Loop over vx, vy, vz
        v_component = velocity_field[component_id]
        vk = pyfftw.builders.rfftn(v_component, auto_align_input=False, auto_contiguous=False, avoid_copy=True)
        vk_cut = vk()[mask]
        vk_cut = vk_cut.reshape(n_grid_target, n_grid_target, int(n_grid_target/2)+1)
        v_downsampled = pyfftw.builders.irfftn(vk_cut, axes=(0,1,2))()
        velocity_field_kcut.append(v_downsampled)

    velocity_field_kcut = np.array(velocity_field_kcut)
    return velocity_field_kcut


def round_to_nearest_even(x):
    """Round a number to the nearest even integer (needed for FFTs)."""
    return int(round(x / 2) * 2)

__all__ = [
    "param_label_dict",
    "cosmo_dict_quijote",
    "cosmo_dict_shame",
    "bias_dict_shame",
    "cosmo_param_names_ordered",
    "biasparam_names_ordered",
    "param_names_all_ordered",
    "noiseparam_names_ordered",
    "setup_cosmo_emu",
    "load_emu",
    "param_dict_to_bacco_param_dict",
    "get_cosmo",
    "get_cosmo_emu",
    "cosmo_bacco_to_cosmo_baccoemu",
    "param_name_to_param_name_emu",
    "get_tracer_field",
    "pnn_to_pk",
    "pnn_to_pgm",
    "remove_highk_modes",
    "remove_highk_modes_velocity",
    "round_to_nearest_even",
]

