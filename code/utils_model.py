"""Cosmology/model-related helpers and parameter dictionaries."""

import numpy as np

# Quijote: https://arxiv.org/pdf/1909.05273, Table 1, top row
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

cosmo_dict_shame = {
            'omega_cdm'     : 0.2603,
            'omega_baryon'  : 0.0486,
            'omega_cold'    : 0.30889999999999995,
            'omega_m'       : 0.30889999999999995,
            'neutrino_mass' : 0.0,
            'hubble'        : 0.6774,
            'ns'            : 0.9667,
            'As'            : 2.0669036404058968e-09,
            'sigma8_cold'   : 0.8159,
            'sigma8'        : 0.8159000277519226,
            'w0'            : -1.0,
            'wa'            : 0.0,
            'tau'           : 0.0952,
}

bias_dict_shame = {
    '_nbar0.00011': {'b1': 0.52922445, 'b2': 0.13816352*2, 'bs2': -0.21806094*2, 'bl': -1.0702721},
    '_nbar0.00022': {'b1': 0.47410742, 'b2': 0.06350746*2, 'bs2': -0.16940883*2, 'bl': -0.82443643},
    '_nbar0.00054': {'b1': 0.40209658, 'b2': -0.00958755*2, 'bs2': -0.09669132*2, 'bl': -0.79150708},
    '_nbar0.00011_bl0': {'b1': 0.52922445, 'b2': 0.13816352*2, 'bs2': -0.21806094*2, 'bl': 0.0},
    '_nbar0.00022_bl0': {'b1': 0.47410742, 'b2': 0.06350746*2, 'bs2': -0.16940883*2, 'bl': 0.0},
    '_nbar0.00054_bl0': {'b1': 0.40209658, 'b2': -0.00958755*2, 'bs2': -0.09669132*2, 'bl': 0.0},
}

cosmo_param_names_ordered = ['omega_cold', 'sigma8_cold', 'hubble', 'omega_baryon', 'ns']
biasparam_names_ordered = ['b1', 'b2', 'bs2', 'bl']
param_names_all_ordered = cosmo_param_names_ordered + biasparam_names_ordered
noiseparam_names_ordered = ['An_homog', 'An_b1', 'An_b2', 'An_bs2', 'An_bl']

n_factor_arr = [1, 2, 4, 8, 16, 32]
nest_level_to_n_factor = {i: f for i, f in enumerate(n_factor_arr)}
n_factor_to_nest_level = {v: k for k, v in nest_level_to_n_factor.items()}


def load_emu(emu_name='lbias_2.0', dir_emus_lbias=None, dir_emus_mpk=None):
    import baccoemu
    if dir_emus_lbias is None:
        dir_emus_lbias = '/cosmos_storage/data_sharing/data_share'
    if dir_emus_mpk is None:
        dir_emus_mpk = '/cosmos_storage/data_sharing/datashare'
    if emu_name == 'lbias_public':
        emu = baccoemu.Lbias_expansion(verbose=False)
    elif emu_name == 'lbias_2.0':
        fn_emu = f'{dir_emus_lbias}/lbias_emulator/lbias_emulator2.0.0'
        emu = baccoemu.Lbias_expansion(
            verbose=False,
            nonlinear_emu_path=fn_emu,
            nonlinear_emu_details='details.pickle',
            nonlinear_emu_field_name='NN_n',
            nonlinear_emu_read_rotation=False,
        )
    elif emu_name == 'mpk':
        standardspace_folder = f'{dir_emus_mpk}/mpk_baccoemu_new/mpk_oldsims_standard_emu_npca7_neurons_400_400_dropout_0.0_bn_False/'
        emu = baccoemu.Matter_powerspectrum(
            nonlinear_emu_path=standardspace_folder,
            nonlinear_emu_details='details.pickle',
            verbose=False,
        )
    elif emu_name == 'mpk_extended':
        extendedspace_folder = f'{dir_emus_mpk}/mpk_baccoemu_new/mpk_extended_emu_npca_20_batch_size_256_nodes_400_400_dropout_0.0_batch_norm_False/'
        emu = baccoemu.Matter_powerspectrum(
            nonlinear_emu_path=extendedspace_folder,
            nonlinear_emu_details='details.pickle',
            verbose=False,
        )
    else:
        raise ValueError(f'Emulator {emu_name} not recognized!')
    emu_param_names = emu.emulator['nonlinear']['keys']
    emu_bounds = emu.emulator['nonlinear']['bounds']
    return emu, emu_bounds, emu_param_names


def param_name_to_param_name_emu(param_name):
    """Map bacco param name to emulator name (e.g. sigma8 -> sigma8_cold). Not exact for nonzero neutrino_mass."""
    if param_name == 'sigma8':
        return 'sigma8_cold'
    return param_name


def param_dict_to_bacco_param_dict(param_dict, neutrino_mass=None):
    if neutrino_mass is None:
        assert 'neutrino_mass' in param_dict, "must pass neutrino mass in param dict or separately!"
        neutrino_mass = param_dict['neutrino_mass']
    names_to_bacco = {'h': 'hubble', 'sigma_8': 'sigma8', 'sigma8_cold': 'sigma8', 'n_s': 'ns'}
    param_dict_bacco = {}
    for name_orig in param_dict:
        param_dict_bacco[names_to_bacco.get(name_orig, name_orig)] = param_dict[name_orig]
    if 'omega_cdm' in param_dict_bacco:
        pass
    elif 'omega_cold' in param_dict_bacco:
        param_dict_bacco['omega_cdm'] = param_dict_bacco['omega_cold'] - param_dict_bacco['omega_baryon']
    elif 'omega_m' in param_dict_bacco:
        param_dict_bacco['omega_cold'] = param_dict_bacco['omega_m'] - neutrino_mass
        param_dict_bacco['omega_cdm'] = param_dict_bacco['omega_cold'] - param_dict_bacco['omega_baryon']
    return param_dict_bacco


def get_cosmo(param_dict, a_scale=1, sim_name='quijote'):
    import bacco
    param_names_bacco = ['omega_cdm', 'omega_baryon', 'hubble', 'ns', 'sigma8', 'tau', 'neutrino_mass', 'w0', 'wa']
    if sim_name == 'quijote':
        param_dict_bacco_fid = param_dict_to_bacco_param_dict(cosmo_dict_quijote)
    else:
        raise ValueError(f'Simulation {sim_name} not recognized!')
    neutrino_mass = param_dict['neutrino_mass'] if 'neutrino_mass' in param_dict else param_dict_bacco_fid['neutrino_mass']
    param_dict_bacco = param_dict_to_bacco_param_dict(param_dict, neutrino_mass=neutrino_mass)
    cosmopars = {pn: param_dict_bacco[pn] if pn in param_dict_bacco else param_dict_bacco_fid[pn] for pn in param_names_bacco}
    cosmo = bacco.Cosmology(**cosmopars, verbose=False)
    cosmo.set_expfactor(a_scale)
    return cosmo


def get_tracer_field(bias_fields_eul, bias_vector, n_grid_norm, noise_field=None, A_noise=None, noise_model='multiplicative'):
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
            tracer_field_noise = np.sum([bias_fields_eul[ii] * A_noise[ii] * noise_field for ii in range(len(bias_fields_eul))], axis=0)
            tracer_field_noise /= n_grid_norm**3
        elif noise_model == 'additive':
            assert isinstance(A_noise, (float, int)), "A_noise must be a single number for additive noise"
            tracer_field_noise = A_noise * noise_field
        else:
            raise ValueError(f"Noise type {noise_model} not recognized!")
        tracer_field_eul_norm += tracer_field_noise
    return tracer_field_eul_norm


def pnn_to_pk(pnn, bias_params, return_cross=False, pk_type='pk'):
    message = 'Please, pass a valid bias array, with' + 'b1, b2, bs2, blaplacian'
    assert len(bias_params) == 4, message
    import itertools
    bias_params_extended = np.concatenate(([1], bias_params))
    prod = np.array(list(itertools.combinations_with_replacement(np.arange(len(bias_params_extended)), r=2)))
    pgal_auto = 0
    for i in range(len(pnn)):
        fac = 2 if prod[i, 0] != prod[i, 1] else 1
        pgal_auto += bias_params_extended[prod[i, 0]] * bias_params_extended[prod[i, 1]] * fac * pnn[i][pk_type]
    return pgal_auto


def pnn_to_pgm(pnn, bias_params, pk_type='pk'):
    message = 'Please, pass a valid bias array, with b1, b2, bs2, bl'
    assert len(bias_params) == 4, message
    bias_params_extended = np.concatenate(([1], bias_params))
    pnn_idx_pgm = [1, 5, 6, 7, 8]
    pks_cross = [pnn[i][pk_type] for i in pnn_idx_pgm]
    return np.dot(bias_params_extended, pks_cross)


def remove_highk_modes(field, box_size_mock, n_grid_target):
    import bacco
    import pyfftw
    n_grid = field.shape[-1]
    k_nyq = np.pi / box_size_mock * n_grid_target
    kmesh = bacco.visualization.np_get_kmesh((n_grid, n_grid, n_grid), box_size_mock, real=True)
    mask = (kmesh[:,:,:,0]<=k_nyq) & (kmesh[:,:,:,1]<=k_nyq) & (kmesh[:,:,:,2]<=k_nyq) & (kmesh[:,:,:,0]>-k_nyq) & (kmesh[:,:,:,1]>-k_nyq) & (kmesh[:,:,:,2]>-k_nyq)
    assert n_grid_target % 2 == 0, "n_grid_target must be even!"
    deltak = pyfftw.builders.rfftn(field, auto_align_input=False, auto_contiguous=False, avoid_copy=True)
    deltakcut = deltak()[mask]
    deltakcut = deltakcut.reshape(n_grid_target, n_grid_target, int(n_grid_target/2)+1)
    return pyfftw.builders.irfftn(deltakcut, axes=(0,1,2))()


def remove_highk_modes_velocity(velocity_field, box_size, n_grid_target):
    import bacco
    import pyfftw
    n_grid = velocity_field.shape[-1]
    k_nyq = np.pi / box_size * n_grid_target
    kmesh = bacco.visualization.np_get_kmesh((n_grid, n_grid, n_grid), box_size, real=True)
    mask = (kmesh[:,:,:,0]<=k_nyq) & (kmesh[:,:,:,1]<=k_nyq) & (kmesh[:,:,:,2]<=k_nyq) & (kmesh[:,:,:,0]>-k_nyq) & (kmesh[:,:,:,1]>-k_nyq) & (kmesh[:,:,:,2]>-k_nyq)
    velocity_field_kcut = []
    assert n_grid_target % 2 == 0, "n_grid_target must be even!"
    for component_id in range(3):
        v_component = velocity_field[component_id]
        vk = pyfftw.builders.rfftn(v_component, auto_align_input=False, auto_contiguous=False, avoid_copy=True)
        vk_cut = vk()[mask]
        vk_cut = vk_cut.reshape(n_grid_target, n_grid_target, int(n_grid_target/2)+1)
        velocity_field_kcut.append(pyfftw.builders.irfftn(vk_cut, axes=(0,1,2))())
    return np.array(velocity_field_kcut)


def round_to_nearest_even(x):
    return int(round(x / 2) * 2)
