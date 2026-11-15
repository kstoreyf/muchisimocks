"""select_shame_noisebest.py

Choose the best-fit multiplicative noise amplitudes for the fixed-SHAMe-cosmology test set
(`_biasshame_noisebest_p0_n1`) on the phase-matched muchisimocks fields, instead of picking
the best of 1000 library mocks with different ICs (which partly selected a lucky
cosmic-variance draw; notebooks/2026-06-03_shame_noise_params.ipynb).

Metric (same as the 2026-06-03 selection): joint reduced diagonal chi2,
    chi2 = w_P * chi2_P / n_P + w_B * chi2_B / n_B (+ w_Pgm * ...),  w = (1, 1, 0),
P over all bins, B over triangles with all k_i < 0.25 (`_kb0.25`), against the raw SHAMe
nbar = 2.2e-4 statistics (phase 0/pi pair mean). Biases fixed to bias_dict_shame (the
values the test set uses). Per-bin variance: there is no cosmic variance left in the
residual, so it can't come from the matched mock itself; it stands in for the per-mock
error the old ensemble variance gave:
  - P: Gaussian, 2 P_SHAMe^2 / N_modes
  - B: fractional std per triangle of the old CV ensemble (_shame_p0_n1000 x old noisebest,
       1000 mocks), times (1000/1024)^1.5 for the matched box volume.
Instead of a discrete pick, A is optimized continuously.

Emulator: the tracer t = (O_1 - 1) + sum_i b_i O_i + sum_j A_j eps O_j is linear in
x = (b1, b2, bs2, bl, A_homog, A_b1, A_b2, A_bs2, A_bl), so P is exactly quadratic and B
exactly cubic in x (cubic least-squares fit, checked on held-out points). Coefficients are
averaged over phases and noise realizations (= expectation over eps). Since b is a variable
too, the same coefficients give the free-bias fit (used to check the noise priors).

Amplitudes: the matched grid (1024/132) has 2.1% smaller cells than the library (1000/128),
so A_lib = A_matched / V_SCALE (see compute_shame_matched_noisebest.py).

Usage:
  python select_shame_noisebest.py emulate TASK   # TASK = 0..7 -> (phase, eps realization); Slurm array
  python select_shame_noisebest.py fit            # ~1 min, login node; writes params_fixed + results
Job: code_auxiliary/jobs/job_select_shame_noisebest.s
Output: data/data_shame/matched_seed-406168/noisebest_select/{task*.npy, fit.npy}
        data/params/params_fixed_biasshame_noisebest_p0_n1.txt
"""
import os
import sys
import glob
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

import compute_shame_matched_noisepoly as cnp   # matched-grid P estimator, field loaders, monomials

SEED = cnp.SEED
TAG_MOCK = '_nbar0.00022'
PHASES = ['0', 'pi']
N_REAL = 4
TASKS = [(ph, i) for ph in PHASES for i in range(N_REAL)]
SEED_NOISE_BASE = 40000
N_THREADS = int(os.environ.get('SLURM_NTASKS', 16))
N_DESIGN, N_TEST = 280, 10          # > 220 cubic monomials in 9 variables
KMAX_B = 0.25
PROBE_WEIGHTS = {'pk': 1.0, 'bispec': 1.0, 'pgm': 0.0}
V_SCALE = ((1000. / 128) / (cnp.L / cnp.NG))**1.5
BIAS_NAMES = ['b1', 'b2', 'bs2', 'bl']
NOISE_NAMES = ['An_homog', 'An_b1', 'An_b2', 'An_bs2', 'An_bl']
# design center/scale for the cubic fit (fixed, so coefficients from all tasks can be averaged)
X0 = np.zeros(9)
XS = np.array([0.5, 0.5, 0.5, 2.0, 1.5, 1.5, 1.5, 3.0, 8.0])
DIR_OUT = cnp.msm.DIR_DATA_SHAME / f'matched_seed{SEED}' / 'noisebest_select'
DIR_ENS_B = Path('/scratch/kstoreyf/muchisimocks/data/bispecs_mlib/'
                 'bispecs_shame_p0_n1000_biasshame_noisebest_p0_n1_noise_unit_shame_p0_n1000')
TAG_PARAMS_OUT = '_biasshame_noisebest_p0_n1'


def setup_bk():
    import PolyBin3D as pb
    import compute_statistics as cs
    base = cs.setup_bispec(cnp.L, cnp.NG, n_threads=N_THREADS)
    k_edges = np.linspace(0.01, 0.4, 13)           # production binning (compute_bispectrum)
    bspec = pb.BSpec(base, k_edges, applySinv=None, mask=None, lmax=1,
                     k_bins_squeeze=k_edges.copy(), include_partial_triangles=False)
    return bspec.get_ks(), lambda f: bspec.Bk_ideal(f.astype(np.float64), discreteness_correction=True)['b0']


def emulate(task):
    phase, ireal = TASKS[task]
    k123, Bk = setup_bk()
    ops = cnp.load_ops(phase)
    eps = np.random.default_rng(SEED_NOISE_BASE + ireal).standard_normal(ops[0].shape)
    comps = [ops[0] - 1] + list(ops[1:]) + [eps * o for o in ops]      # c_0, then the 9 x-components
    comps = [c - c.mean() for c in comps]
    comps_k = [cnp.ft(c) for c in comps]
    dm_k = comps_k[0]                                                   # v2 matter = O_1 - 1
    n = len(comps)
    Pmat = np.array([[cnp.pk(comps_k[a], comps_k[c]) for c in range(n)] for a in range(n)])
    Pgm = np.array([cnp.pk(ck, dm_k) for ck in comps_k])

    def tracer(x):
        return comps[0] + sum(xj * c for xj, c in zip(x, comps[1:]))

    expo = cnp.monomials(9)
    U = np.random.default_rng(1000 + task).standard_normal((N_DESIGN + N_TEST, 9))
    t0 = time.time()
    Bs = np.array([Bk(tracer(X0 + XS * u)) for u in U])
    coef, *_ = np.linalg.lstsq(cnp.design(U[:N_DESIGN], expo), Bs[:N_DESIGN], rcond=None)
    err = np.abs(cnp.design(U[N_DESIGN:], expo) @ coef / Bs[N_DESIGN:] - 1).max()
    print(f'{phase=} {ireal=}: {len(U)} B evals in {time.time() - t0:.0f} s; held-out max|B_poly/B-1| = {err:.1e}', flush=True)

    out = dict(Pmat=Pmat, Pgm=Pgm, Bcoef=coef, expo=expo, B_holdout_err=err, k_pk=cnp.k_pk, k123=k123)
    if ireal == 0:
        tr = cnp.load_shame_tracer(phase)
        tr_k = cnp.ft(tr)
        out['shame'] = dict(pk=cnp.pk(tr_k, tr_k), pgm=cnp.pk(tr_k, dm_k), bk=Bk(tr))
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    np.save(DIR_OUT / f'task{task}.npy', out)


# ---------------------------------------------------------------- fit
def load_tasks():
    t = {i: np.load(DIR_OUT / f'task{i}.npy', allow_pickle=True).item() for i in range(len(TASKS))}
    sh = {s: np.mean([t[i]['shame'][s] for i in t if 'shame' in t[i]], axis=0) for s in ['pk', 'pgm', 'bk']}
    return t, sh


def coeffs(t, idx):
    return dict(Pmat=np.mean([t[i]['Pmat'] for i in idx], axis=0),
                Pgm=np.mean([t[i]['Pgm'] for i in idx], axis=0),
                Bcoef=np.mean([t[i]['Bcoef'] for i in idx], axis=0))


def make_stats(expo):
    def stats(x, C):
        a = np.concatenate([[1.], x])
        u = (x - X0) / XS
        return dict(pk=np.einsum('a,c,ack->k', a, a, C['Pmat']), pgm=a @ C['Pgm'],
                    bk=np.prod(u[None, :] ** expo, axis=1) @ C['Bcoef'])
    return stats


def bispec_frac_std():
    bs = np.array([np.load(f, allow_pickle=True).item()['bispectrum']['b0']
                   for f in sorted(glob.glob(str(DIR_ENS_B / 'bispec_*.npy')))])
    return bs.std(axis=0) / np.abs(bs.mean(axis=0)) * (1000. / 1024)**1.5, len(bs)


def fit():
    from scipy.optimize import minimize
    import utils_model
    import generate_params as genp

    t, sh = load_tasks()
    t0 = t[0]
    stats = make_stats(t0['expo'])
    k, k123 = t0['k_pk'], t0['k123']
    mP = np.ones(len(k), bool)
    mB = k123.max(axis=0) < KMAX_B
    sig_frac_B, n_ens = bispec_frac_std()
    var = dict(pk=2 * sh['pk']**2 / cnp.nmode, bk=(sig_frac_B * sh['bk'])**2,
               pgm=(sh['pk'] * sh['pk'] + sh['pgm']**2) / cnp.nmode)   # pgm: rough Gaussian, weight 0 anyway
    masks = dict(pk=mP, bk=mB, pgm=mP)
    wkey = dict(pk='pk', bk='bispec', pgm='pgm')

    def chi2_red(x, C):
        s = stats(x, C)
        return {p: np.mean((s[p][m] - sh[p][m])**2 / var[p][m]) for p, m in masks.items()}

    def loss(x, C):
        c = chi2_red(x, C)
        return sum(PROBE_WEIGHTS[wkey[p]] * c[p] for p in c)

    b_shame = np.array([utils_model.bias_dict_shame[TAG_MOCK][n] for n in BIAS_NAMES])
    bounds_A = [tuple(genp.BOUNDS['Anoisemult'][n]) for n in NOISE_NAMES]
    bounds_b = [tuple(genp.BOUNDS['bias'][n]) for n in BIAS_NAMES]
    fn_params = HERE.parent / 'data' / 'params' / f'params_fixed{TAG_PARAMS_OUT}.txt'
    old = np.genfromtxt(fn_params, delimiter=',', names=True)
    A_old_matched = np.array([float(old[n]) for n in NOISE_NAMES]) * V_SCALE

    def optimize(C, free_b, bounded, starts):
        def f(z):
            x = np.concatenate([z[:4], z[4:]]) if free_b else np.concatenate([b_shame, z])
            return loss(x, C)
        bnds = ((bounds_b if free_b else []) + bounds_A) if bounded else None
        res = [minimize(f, z0, method='L-BFGS-B', bounds=bnds) for z0 in starts]
        best = min(res, key=lambda r: r.fun)
        x = best.x if free_b else np.concatenate([b_shame, best.x])
        return x, best.fun

    rng = np.random.default_rng(0)
    C_all = coeffs(t, range(len(TASKS)))
    starts_A = [A_old_matched] + [rng.uniform(*np.array(bounds_A).T) for _ in range(15)]
    starts_bA = [np.concatenate([b_shame, a]) for a in starts_A]
    out = dict(k_pk=k, k123=k123, mB=mB, var=var, sh=sh, n_ens_B=n_ens, V_scale=V_SCALE,
               bias_names=BIAS_NAMES, noise_names=NOISE_NAMES, weights=PROBE_WEIGHTS,
               bounds_A=dict(zip(NOISE_NAMES, bounds_A)), fits={})
    fits = {'old noisebest (LH pick)': (np.concatenate([b_shame, A_old_matched]), None),
            'SHAMe b, no noise': (np.concatenate([b_shame, np.zeros(5)]), None)}
    for label, free_b, bounded in [('SHAMe b, A fit (prior-bounded)', False, True),
                                   ('SHAMe b, A fit (unbounded)', False, False),
                                   ('b + A fit (prior-bounded)', True, True),
                                   ('b + A fit (unbounded)', True, False)]:
        fits[label] = optimize(C_all, free_b, bounded, starts_bA if free_b else starts_A)
    for label, (x, _) in fits.items():
        c = chi2_red(x, C_all)
        s = stats(x, C_all)
        out['fits'][label] = dict(x=x, A_lib=x[4:] / V_SCALE, chi2_red=c, loss=loss(x, C_all), stats=s)
        print(f'{label:32s} loss={loss(x, C_all):7.3f}  chi2_red P={c["pk"]:.2f} B={c["bk"]:.2f} Pgm={c["pgm"]:.2f}  '
              f'b={np.round(x[:4], 3)}  A_lib={np.round(x[4:] / V_SCALE, 3)}')

    # held-out eps: fit on realizations 0,1, evaluate on 2,3 (both phases)
    C_a = coeffs(t, [i for i, (ph, r) in enumerate(TASKS) if r < 2])
    C_b = coeffs(t, [i for i, (ph, r) in enumerate(TASKS) if r >= 2])
    x_a, _ = optimize(C_a, False, True, starts_A)
    out['holdout'] = (loss(x_a, C_a), loss(x_a, C_b), x_a[4:] / V_SCALE)
    print(f'held-out eps (SHAMe b, bounded): loss {loss(x_a, C_a):.3f} -> {loss(x_a, C_b):.3f}; A_lib={np.round(x_a[4:] / V_SCALE, 3)}')

    sel = out['fits']['SHAMe b, A fit (prior-bounded)']
    at_bound = [n for n, a, (lo, hi) in zip(NOISE_NAMES, sel['x'][4:], bounds_A) if min(a - lo, hi - a) < 1e-3 * (hi - lo)]
    out['selected'] = dict(zip(BIAS_NAMES + NOISE_NAMES, map(float, np.concatenate([b_shame, sel['A_lib']]))))
    out['old'] = dict(zip(NOISE_NAMES, A_old_matched / V_SCALE))
    out['at_bound'] = at_bound
    if at_bound:
        print('WARNING: selected amplitudes at prior bound:', at_bound)
    np.save(DIR_OUT / 'fit.npy', out)
    genp.write_authored_params_fixed(TAG_PARAMS_OUT, out['selected'], param_names=BIAS_NAMES + NOISE_NAMES)
    print(f'Wrote {fn_params}\nSaved {DIR_OUT / "fit.npy"}')


if __name__ == '__main__':
    if sys.argv[1] == 'emulate':
        emulate(int(sys.argv[2]))
    elif sys.argv[1] == 'fit':
        fit()
    else:
        raise ValueError(sys.argv[1])
