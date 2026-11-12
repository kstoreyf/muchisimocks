"""make_shame_matched_mock.py

Generate muchisimocks bias fields with the SAME initial phases (cosmic variance)
as the SHAMe high-fidelity OOD mock, so muchisimocks-vs-SHAMe comparisons are
free of sample variance.

The SHAMe mock lives on the BACCO "Rings" paired-and-fixed run
    Planck_N3072_L1024.0_output/{0.00, 3.14}
i.e. FixedInitialAmplitude=True, InitialPhase in {0, pi}, BoxSize=1024 Mpc/h.

Rather than the new bacco map2map interface (not in our installed bacco), this
reuses code/data_creation_pipeline.run_single (ZA -> map2map disp+vel -> bias
model -> k-cut -> deconvolve) with:
  - box_size = 1024           (so the Fourier modes / phases match the sim exactly)
  - n_grid   = 524            (= 1024 / (1000/512), keeps map2map's cell size ~1.95 Mpc/h)
  - n_grid_target = 132       (= 1024 / (1000/128), same as process_mock_catalogs.py
                               uses for the SHAMe tracer/density meshes)
  - FixedInitialAmplitude=True, InitialPhase=0 or pi, ngenic_phases=True, phase_type=1
These grid choices match round_to_nearest_even() in process_mock_catalogs.py, so
outputs are directly comparable to data/data_shame/*.

N-GenIC phases are resolution-independent at low k, so a 524^3 (or 132^3) IC with
the right seed reproduces the large-scale modes of the 3072^3 sim.

!! The SHAMe/Rings seed is NOT recorded anywhere in this repo or in the catalog
   files. Ask for it (or read it from the sim's Gadget param file on atlas), then
   verify with `check_seed` before spending GPU time on `make`.

Must run on a compute node (bacco's IC generation dies silently on the login node).

Usage:
  # 1. Cheap sanity check (~CPU only): cross-correlate the linear field for a
  #    candidate seed with the SHAMe z=0 matter mesh. Right seed -> r(k)~1 at low k;
  #    wrong seed -> r(k)~0. Also tries phase_type 1/2 and axis permutations.
  python make_shame_matched_mock.py check_seed --seeds 424 1234

  # 2. Full mocks (needs GPU for map2map), both paired phases:
  python make_shame_matched_mock.py make --seed <SEED>
  python make_shame_matched_mock.py make --seed <SEED> --phases 0      # just one
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paths
import utils_model

BOX_SIZE = 1024.0
BOX_SIZE_MLIB = 1000.0
N_GRID = utils_model.round_to_nearest_even(BOX_SIZE / (BOX_SIZE_MLIB / 512))         # 524
N_GRID_TARGET = utils_model.round_to_nearest_even(BOX_SIZE / (BOX_SIZE_MLIB / 128))  # 132

PHASES = {'0': 0.0, 'pi': np.pi}
DIR_DATA_SHAME = HERE.parent / 'data' / 'data_shame'
SUBDIR_OOD = 'shame_matched'


# ----------------------------------------------------------------------
# Seed check
# ----------------------------------------------------------------------
def cross_corr_coeff(field_a, field_b, box_size, n_bins=12, k_max=0.15):
    """r(k) = P_ab / sqrt(P_aa P_bb), binned in |k| (h/Mpc)."""
    n = field_a.shape[0]
    fa = np.fft.rfftn(field_a - field_a.mean())
    fb = np.fft.rfftn(field_b - field_b.mean())
    kf = 2 * np.pi / box_size
    kx = np.fft.fftfreq(n, d=1. / n) * kf
    kz = np.fft.rfftfreq(n, d=1. / n) * kf
    kmag = np.sqrt(kx[:, None, None]**2 + kx[None, :, None]**2 + kz[None, None, :]**2)
    edges = np.linspace(kf, k_max, n_bins + 1)
    idx = np.digitize(kmag.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < n_bins)
    idx = idx[ok]
    pab = np.bincount(idx, (fa * np.conj(fb)).real.ravel()[ok], n_bins)
    paa = np.bincount(idx, (np.abs(fa)**2).ravel()[ok], n_bins)
    pbb = np.bincount(idx, (np.abs(fb)**2).ravel()[ok], n_bins)
    return 0.5 * (edges[1:] + edges[:-1]), pab / np.sqrt(paa * pbb)


def linear_field(seed, phase, phase_type=1, n_grid=N_GRID_TARGET):
    import bacco
    bacco.configuration.update({'pk': {'boltzmann_solver': 'CLASS'}})
    bacco.configuration.update({'pknbody': {'ngrid': n_grid}})
    bacco.configuration.update({'scaling': {'disp_ngrid': n_grid}})
    cosmo = utils_model.get_cosmo(utils_model.cosmo_dict_shame, a_scale=1.0)
    sim, _ = bacco.utils.create_lpt_simulation(
        cosmo, BOX_SIZE, Nmesh=n_grid, Seed=seed,
        FixedInitialAmplitude=True, InitialPhase=phase,
        expfactor=1.0, LPT_order=1, order_by_order=None,
        phase_type=phase_type, ngenic_phases=True, return_disp=True, sphere_mode=0,
    )
    return np.asarray(sim.linear_field[0], dtype=np.float64)


def check_seed(seeds, phase_types=(1, 2), phase_name='0'):
    fn_dens = DIR_DATA_SHAME / f'dens_mesh_phase{phase_name}.npy'
    dens = np.load(fn_dens)
    assert dens.shape == (N_GRID_TARGET,) * 3, f'{fn_dens} has shape {dens.shape}'
    print(f'Comparing against {fn_dens} (shape {dens.shape})')
    for seed in seeds:
        for pt in phase_types:
            lin = linear_field(seed, PHASES[phase_name], phase_type=pt)
            # try axis permutations in case mesh index conventions differ
            best = None
            for perm in itertools.permutations(range(3)):
                k, r = cross_corr_coeff(lin.transpose(perm), dens, BOX_SIZE)
                score = np.mean(r[:4])
                if best is None or abs(score) > abs(best[0]):
                    best = (score, perm, r)
            score, perm, r = best
            print(f'seed={seed} phase_type={pt} best_perm={perm} '
                  f'mean r(k<{k[3]:.3f})={score:+.3f}  r(k)={np.round(r, 2)}', flush=True)
    print('Expect r ~ +0.95-1 at the lowest k for the right seed (identity perm), '
          '~0 +/- 0.1 for a wrong one; r ~ -1 means the 0/pi phases are swapped.')


# ----------------------------------------------------------------------
# Full mock
# ----------------------------------------------------------------------
def make(seed, phase_names):
    import data_creation_pipeline
    dir_mocks = paths.mocks_lib_ood_dir(SUBDIR_OOD)
    for name in phase_names:
        print(f'=== SHAMe-matched mock: seed={seed}, phase={name}, '
              f'box={BOX_SIZE}, n_grid={N_GRID}, n_grid_target={N_GRID_TARGET} ===', flush=True)
        # -> {dir_mocks}/phase{name}_seed{seed}/bias_fields_eul[_zspace]_deconvolved_{seed}.npy
        data_creation_pipeline.run_single(
            dict(utils_model.cosmo_dict_shame), seed, seed, str(dir_mocks),
            n_grid=N_GRID, n_grid_target=N_GRID_TARGET, box_size=BOX_SIZE,
            n_threads_bacco=8, n_workers_m2m=0,
            subdir_prefix=f'phase{name}_seed',
            fixed_amplitude=True, initial_phase=PHASES[name],
        )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    pc = sub.add_parser('check_seed')
    pc.add_argument('--seeds', type=int, nargs='+', required=True)
    pc.add_argument('--phase_types', type=int, nargs='+', default=[1, 2])
    pc.add_argument('--phase', choices=list(PHASES), default='0')
    pm = sub.add_parser('make')
    pm.add_argument('--seed', type=int, required=True)
    pm.add_argument('--phases', nargs='+', choices=list(PHASES), default=list(PHASES))
    args = p.parse_args()

    if args.cmd == 'check_seed':
        check_seed(args.seeds, args.phase_types, args.phase)
    else:
        make(args.seed, args.phases)


if __name__ == '__main__':
    main()
