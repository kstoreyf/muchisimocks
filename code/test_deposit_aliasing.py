"""test_deposit_aliasing.py

Empirical test of aliasing / CIC-deconvolution error in the Eulerian deposit step
of data_creation_pipeline.predicted_positions_to_bias_fields.

The Lagrangian operators (1, delta, delta^2, s^2, nabla^2 delta) are band-limited
by the damping_scale=0.75 h/Mpc cut, so building them on 512^3 is alias-free
(Orszag). But the deposit (CIC of 512^3 weighted particles at the nonlinear
map2map positions onto a mesh) is NOT band-limited, so it can alias power from
k > k_N into our k < 0.4 band, and dividing by the CIC window can't remove that.

Test: keep the 512^3 particles + Lagrangian weights fixed and vary ONLY the
deposit mesh (compute_mesh ngrid), then k-cut to 128^3 and deconvolve with the
matching npix, exactly as the pipeline does:

    ng256   k_N = 0.80 h/Mpc   (expect visible aliasing; shows the test is sensitive)
    ng512   k_N = 1.61         (production)
    ng512i  k_N = 1.61, interlacing=True
    ng1024  k_N = 3.22         (reference)

Input: a mock with intermediates saved, i.e. muchisimocks_lib_p5_n10000_rerun/LH0
(pred_disp.npy). The Lagrangian fields are rebuilt by regenerating the ZA sim
from the same seed/cosmology, as in data_creation_pipeline.run_single
(BiasModel(sim=...) is the codepath that applies the damping).

Validation built in:
  - regenerated linear field vs saved lin_field.npy
  - ng512 vs the production bias_fields_eul_deconvolved_<idx>.npy (should be ~identical)
  - optional --dir_noise_floor (e.g. LHagain0): production-vs-production difference
    from map2map nondeterminism, a "differences below this don't matter" yardstick

Outputs (in paths.DIR_DATA/aliasing_test/<LHdir>/):
  bias_fields_<cfg>.npy   (5,128,128,128) k-cut, deconvolved, like production
  summary.npz             k, auto P(k) per operator/config, tracer P(k), diff power

Must run on a compute node (bacco IC generation dies on the login node).
ng1024 peaks at ~20-25 GB; ask for 64G. No GPU needed.

  python test_deposit_aliasing.py
  python test_deposit_aliasing.py --configs ng512 ng1024      # subset
  python test_deposit_aliasing.py --analyze_only              # re-run summary from saved fields
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfftw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paths
import utils_model

BOX_SIZE = 1000.
N_GRID = 512           # particles / Lagrangian grid (fixed)
N_GRID_TARGET = 128
DAMPING_SCALE = 0.75
CONFIGS = {            # name: (deposit ngrid, interlacing)
    'ng256': (256, False),
    'ng512': (512, False),
    'ng512i': (512, True),
    'ng1024': (1024, False),
}
REF = 'ng1024'
OP_NAMES = ['1', 'delta', 'delta2', 's2', 'nabla2delta']  # bacco 'expansion' order
TAG_BIAS = '_nbar0.00022'  # SHAMe-like bias values for the tracer P(k)
K_PRINT = [0.05, 0.1, 0.2, 0.3, 0.35, 0.4]

DIR_RERUN = Path('/data/kstoreyf/muchisimocks/muchisimocks_lib_p5_n10000_rerun')
N_THREADS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.environ.get('SLURM_CPUS_ON_NODE', 8)))


# ----------------------------------------------------------------------
# Field operations
# ----------------------------------------------------------------------
def kcut(field, n_target=N_GRID_TARGET, threads=N_THREADS):
    """Same result as utils_model.remove_highk_modes (keeps modes n in (-n_t/2, n_t/2]
    on each axis), but via index slicing so it doesn't build a float64 kmesh
    (~13 GB at 1024^3)."""
    n = field.shape[0]
    h = n_target // 2
    fk = pyfftw.builders.rfftn(field, threads=threads)()
    idx = np.r_[0:h + 1, n - h + 1:n]
    fk_cut = fk[idx][:, idx][:, :, :h + 1]
    del fk
    return pyfftw.builders.irfftn(fk_cut, s=(n_target,) * 3, axes=(0, 1, 2), threads=threads)()


def interlace(f0, f1, ngrid_deposit, box=BOX_SIZE):
    """Combine a k-cut grid f0 and the k-cut grid f1 deposited from positions shifted
    by +d = (H/2)(1,1,1), H = box/ngrid_deposit: F = (F0 + F1 e^{+i k.d}) / 2.
    The odd aliasing images cancel. k-cutting preserves mode indices, so this can
    be done on the small grid."""
    n = f0.shape[0]
    kf = 2 * np.pi / box
    kx = np.fft.fftfreq(n, d=1. / n) * kf
    kz = np.fft.rfftfreq(n, d=1. / n) * kf
    d = 0.5 * box / ngrid_deposit
    phase = np.exp(1j * d * (kx[:, None, None] + kx[None, :, None] + kz[None, None, :]))
    fk = 0.5 * (np.fft.rfftn(f0) + np.fft.rfftn(f1) * phase)
    return np.fft.irfftn(fk, s=f0.shape).astype(np.float32)


def binned_power(a, b=None, box=BOX_SIZE, dk_factor=2):
    """Isotropic P_ab(k) on a small grid, linear bins of width dk_factor*k_f."""
    n = a.shape[0]
    fa = np.fft.rfftn(a)
    fb = fa if b is None else np.fft.rfftn(b)
    kf = 2 * np.pi / box
    kx = np.fft.fftfreq(n, d=1. / n) * kf
    kz = np.fft.rfftfreq(n, d=1. / n) * kf
    kmag = np.sqrt(kx[:, None, None]**2 + kx[None, :, None]**2 + kz[None, None, :]**2)
    # rfft half-space: interior kz planes stand for two modes
    w = np.full(kz.shape, 2.)
    w[0] = 1.
    if n % 2 == 0:
        w[-1] = 1.
    w = np.broadcast_to(w[None, None, :], kmag.shape)
    edges = np.arange(0.5 * kf, np.pi * n / box + dk_factor * kf, dk_factor * kf)
    idx = np.digitize(kmag.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    nb = len(edges) - 1
    wsum = np.bincount(idx[ok], w.ravel()[ok], nb)
    psum = np.bincount(idx[ok], (w * (fa * np.conj(fb)).real).ravel()[ok], nb)
    ksum = np.bincount(idx[ok], (w * kmag).ravel()[ok], nb)
    good = wsum > 0
    norm = box**3 / n**6
    return ksum[good] / wsum[good], psum[good] / wsum[good] * norm


# ----------------------------------------------------------------------
# Rebuild pipeline inputs
# ----------------------------------------------------------------------
def load_param_dict(dir_lib, idx):
    tag = '_p5_n10000'
    pd_fixed = pd.read_csv(dir_lib / f'params_fixed{tag}.txt').loc[0].to_dict()
    param_dict = pd.read_csv(dir_lib / f'params_lh{tag}.txt').loc[idx].to_dict()
    param_dict.update(pd_fixed)
    return param_dict


def lagrangian_fields(dir_LH, idx, seed):
    """Regenerate the ZA sim exactly as run_single does, check it against the saved
    linear field, and return the 5 damped Lagrangian operator fields at 512^3."""
    import bacco
    bacco.configuration.update({'pk': {'boltzmann_solver': 'CLASS'}})
    bacco.configuration.update({'pknbody': {'ngrid': N_GRID}})
    bacco.configuration.update({'scaling': {'disp_ngrid': N_GRID}})
    bacco.configuration.update({'number_of_threads': N_THREADS})

    param_dict = load_param_dict(dir_LH.parent, idx)
    print('param_dict:', param_dict, flush=True)
    # sanity check against the m2m style file saved in the LH dir
    pars_m2m = np.loadtxt(dir_LH / 'cosmo_pars_m2m.txt')
    names_m2m = ['omega_cold', 'omega_baryon', 'hubble', 'ns', 'sigma8_cold']
    assert np.allclose(pars_m2m, [param_dict[p] for p in names_m2m]), 'cosmology mismatch with cosmo_pars_m2m.txt'
    cosmo = utils_model.get_cosmo(param_dict, a_scale=1.0, sim_name='quijote')

    sim, _ = bacco.utils.create_lpt_simulation(
        cosmo, BOX_SIZE, Nmesh=N_GRID, Seed=seed,
        FixedInitialAmplitude=False, InitialPhase=0,
        expfactor=1.0, LPT_order=1, order_by_order=None,
        phase_type=1, ngenic_phases=True, return_disp=True, sphere_mode=0)

    fn_lin = dir_LH / 'lin_field.npy'
    if fn_lin.exists():
        lin_saved = np.load(fn_lin)
        lin_new = sim.linear_field[0] * N_GRID**3.
        rel = np.abs(lin_new - lin_saved).max() / np.abs(lin_saved).max()
        print(f'[check] regenerated vs saved linear field: max rel diff = {rel:.2e}', flush=True)
        assert rel < 1e-4, 'regenerated linear field does not match lin_field.npy; wrong seed/cosmology?'
        del lin_saved, lin_new

    bmodel = bacco.BiasModel(sim=sim, ngrid=N_GRID, ngrid1=None, sdm=False, mode='dm',
                             npart_for_fake_sim=N_GRID, damping_scale=DAMPING_SCALE,
                             bias_model='expansion', deposit_method='cic',
                             use_displacement_of_nn=False, interlacing=False)
    fields = np.asarray(bmodel.bias_terms_lag(), dtype=np.float32)
    del bmodel, sim
    assert fields.shape == (5, N_GRID, N_GRID, N_GRID), fields.shape
    return fields


def predicted_positions(dir_LH):
    import bacco
    pred_disp = np.load(dir_LH / 'pred_disp.npy')
    grid = bacco.visualization.uniform_grid(npix=N_GRID, L=BOX_SIZE, ndim=3, bounds=False)
    pos = bacco.scaler.add_displacement(None, pred_disp, box=BOX_SIZE, pos=grid.reshape(-1, 3),
                                        vel=None, vel_factor=0, verbose=False)[0]
    del pred_disp, grid
    return pos


def deposit_kcut_deconvolve(pos, lag_fields, ngrid, interlacing):
    import bacco
    import data_creation_pipeline
    out = []
    for ii, lf in enumerate(lag_fields):
        t0 = time.time()
        mesh = bacco.statistics.compute_mesh(ngrid=ngrid, box=BOX_SIZE, pos=pos, mass=lf.flatten(),
                                             deposit_method='cic', interlacing=interlacing)
        mesh = np.squeeze(np.asarray(mesh))
        if interlacing:
            # compute_mesh returns [grid, grid of positions shifted by +H/2]
            # (powermodule.c: pos + ig*dhalf); combine after the k-cut
            assert mesh.shape == (2,) + (ngrid,) * 3, mesh.shape
            out.append(interlace(kcut(mesh[0]), kcut(mesh[1]), ngrid))
        else:
            assert mesh.shape == (ngrid,) * 3, mesh.shape
            out.append(kcut(mesh))
        del mesh
        print(f'    op {OP_NAMES[ii]}: {time.time() - t0:.1f}s', flush=True)
    out = np.array(out)
    return data_creation_pipeline.deconvolve_bias_field(out, ngrid).astype(np.float32)


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
def summarize(fields_by_cfg, dir_out, extra=None):
    """fields_by_cfg: {name: (5,128,128,128)}. extra: {name: fields} compared to ng512
    rather than the reference (production check, noise floor)."""
    bias = utils_model.bias_dict_shame[TAG_BIAS]
    bias_vector = [bias[n] for n in utils_model.biasparam_names_ordered]
    tracer = {c: utils_model.get_tracer_field(f, bias_vector, N_GRID) for c, f in fields_by_cfg.items()}

    res = {}
    k, _ = binned_power(tracer[REF])
    res['k'] = k
    for c, f in fields_by_cfg.items():
        res[f'pk_tracer_{c}'] = binned_power(tracer[c])[1]
        res[f'pdiff_tracer_{c}'] = binned_power(tracer[c] - tracer[REF])[1]
        for i, op in enumerate(OP_NAMES):
            res[f'pk_{op}_{c}'] = binned_power(f[i])[1]
            res[f'pdiff_{op}_{c}'] = binned_power(f[i] - fields_by_cfg[REF][i])[1]
    ik = [np.argmin(np.abs(k - kp)) for kp in K_PRINT]

    def row(label, arr):
        return f'{label:<28s}' + ''.join(f'{arr[i]:+10.2e}' for i in ik)

    header = f'{"":<28s}' + ''.join(f'  k={k[i]:.3f}' for i in ik)
    print('\n=== P(k)/P_ref(k) - 1   (ref = %s) ===' % REF)
    print(header)
    for c in fields_by_cfg:
        if c == REF:
            continue
        print(row(f'{c} tracer', res[f'pk_tracer_{c}'] / res[f'pk_tracer_{REF}'] - 1))
        for op in OP_NAMES:
            print(row(f'{c} {op}', res[f'pk_{op}_{c}'] / res[f'pk_{op}_{REF}'] - 1))

    print('\n=== field-level: P(f - f_ref)/P(f_ref)  (~ 2(1-r) + (ratio-1)^2) ===')
    print(header)
    for c in fields_by_cfg:
        if c == REF:
            continue
        print(row(f'{c} tracer', res[f'pdiff_tracer_{c}'] / res[f'pk_tracer_{REF}']))
        for op in OP_NAMES:
            print(row(f'{c} {op}', res[f'pdiff_{op}_{c}'] / res[f'pk_{op}_{REF}']))

    if extra and 'ng512' in fields_by_cfg:
        print('\n=== checks vs ng512: P(f - f_ng512)/P(f_ng512), tracer ===')
        print(header)
        for name, f in extra.items():
            t = utils_model.get_tracer_field(f, bias_vector, N_GRID)
            pdiff = binned_power(t - tracer['ng512'])[1] / res['pk_tracer_ng512']
            res[f'pdiff_tracer_{name}_vs_ng512'] = pdiff
            print(row(name, pdiff))
        print('  production should be ~0 (same code path); noise_floor = map2map nondeterminism')

    np.savez(dir_out / 'summary.npz', **res)
    print(f'\nSaved {dir_out / "summary.npz"}')


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dir_LH', type=Path, default=DIR_RERUN / 'LH0')
    p.add_argument('--idx', type=int, default=0, help='row in params_lh file (= idx_mock)')
    p.add_argument('--seed', type=int, default=None, help='default: idx (as in run_creation)')
    p.add_argument('--configs', nargs='+', default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument('--dir_noise_floor', type=Path, default=DIR_RERUN / 'LHagain0',
                   help='other LH dir from the same params, for a map2map-nondeterminism yardstick')
    p.add_argument('--analyze_only', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    seed = args.idx if args.seed is None else args.seed
    if REF not in args.configs:
        args.configs.append(REF)

    dir_out = paths.DIR_DATA / 'aliasing_test' / args.dir_LH.name
    dir_out.mkdir(parents=True, exist_ok=True)
    fn_cfg = {c: dir_out / f'bias_fields_{c}.npy' for c in args.configs}
    print(f'dir_LH={args.dir_LH}  idx={args.idx}  seed={seed}  out={dir_out}  threads={N_THREADS}', flush=True)

    todo = [c for c in args.configs if args.overwrite or not fn_cfg[c].exists()]
    if todo and not args.analyze_only:
        t0 = time.time()
        lag = lagrangian_fields(args.dir_LH, args.idx, seed)
        print(f'Lagrangian fields: {time.time() - t0:.0f}s', flush=True)
        pos = predicted_positions(args.dir_LH)
        for c in todo:
            ngrid, inter = CONFIGS[c]
            print(f'[{c}] deposit ngrid={ngrid}, interlacing={inter}', flush=True)
            np.save(fn_cfg[c], deposit_kcut_deconvolve(pos, lag, ngrid, inter))
        del lag, pos

    fields = {c: np.load(fn_cfg[c]) for c in args.configs if fn_cfg[c].exists()}
    extra = {}
    fn_prod = args.dir_LH / f'bias_fields_eul_deconvolved_{args.idx}.npy'
    if fn_prod.exists():
        extra['production'] = np.load(fn_prod)
    fn_nf = args.dir_noise_floor / f'bias_fields_eul_deconvolved_{args.idx}.npy' if args.dir_noise_floor else None
    if fn_nf is not None and fn_nf.exists():
        extra['noise_floor'] = np.load(fn_nf)
    summarize(fields, dir_out, extra)


if __name__ == '__main__':
    main()
