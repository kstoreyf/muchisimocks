"""select_shame_noisebest.py

Choose the best-fit multiplicative noise amplitudes for the fixed-SHAMe-cosmology test set
(`_biasshame_noisebest_p0_n1`) on the phase-matched muchisimocks fields, instead of picking
the best of 1000 library mocks with different ICs (which partly selected a lucky
cosmic-variance draw; notebooks/2026-06-03_shame_noise_params.ipynb).

Metric (same as the 2026-06-03 selection): joint reduced diagonal chi2,
    chi2 = w_P * chi2_P / n_P + w_B * chi2_B / n_B (+ w_Pgm * ...),  w = (1, 1, 0),
P over all bins, B over triangles with all k_i < 0.25 (`_kb0.25`), against the raw SHAMe
statistics (phase 0/pi pair mean), for each SHAMe density and bias setting (see Fits below). Per-bin variance (diagonal), Gaussian from the SHAMe total power
(shot noise included), standing in for the per-mock error the old ensemble variance gave:
  - P: 2 P_SHAMe^2 / N_modes
  - B: diag of F^-1, F = PolyBin3D ideal Fisher matrix with P_fid = P_SHAMe (for a Gaussian
       field with power P_fid, F is the inverse covariance of the binned B). Misses the
       non-Gaussian part of the B variance.
Instead of a discrete pick, A is optimized continuously.

Emulator: the tracer t = (O_1 - 1) + sum_i b_i O_i + sum_j A_j eps O_j is linear in
x = (b1, b2, bs2, bl, A_homog, A_b1, A_b2, A_bs2, A_bl), so P is exactly quadratic and B
exactly cubic in x (cubic least-squares fit, checked on held-out points). Coefficients are
averaged over phases and noise realizations (= expectation over eps). Since b is a variable
too, the same coefficients give the free-bias fit (used to check the noise priors).

Amplitudes: the matched grid (1024/132) has 2.1% smaller cells than the library (1000/128),
so A_lib = A_matched / V_SCALE (see compute_shame_matched_noisebest.py).

Usage (Slurm jobs in code_auxiliary/jobs/):
  python select_shame_noisebest.py emulate TASK   # TASK = 0..7 -> (phase, eps); job_select_shame_noisebest.s
  python select_shame_noisebest.py targets        # SHAMe P, B + Gaussian errors, all 3 nbar; job_select_shame_noisebest_targets.s
  python select_shame_noisebest.py fit            # ~1 min; fits every (nbar, bias setting), writes params_fixed; job_select_shame_noisebest_fit.s
Fits: for each SHAMe density, A fit with biases fixed to bias_dict_shame (code convention, b2/bs2 = 2 x raw),
fixed to raw/2 (Stuecker+24 -> our basis), and free; A_homog >= 0, other amplitudes unbounded.
The test-set selection is nbar = 2.2e-4 with the code-convention biases.
Output: data/data_shame/matched_seed-406168/noisebest_select/{task*.npy, targets.npy, fit.npy}
        data/params/params_fixed_biasshame_noisebest_p0_n1.txt
"""
import os
import sys
if len(sys.argv) > 1 and sys.argv[1] == 'fit':   # fit is tiny matmuls
    for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(_v, '1')   # tiny matmuls: threaded BLAS is ~35x slower on a busy node
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

import compute_shame_matched_noisepoly as cnp   # matched-grid P estimator, field loaders, monomials

SEED = cnp.SEED
TAG_MOCK = '_nbar0.00022'                     # density the test set is built for
TAGS_SHAME = ['_nbar0.00011', '_nbar0.00022', '_nbar0.00054']
PHASES = ['0', 'pi']
N_REAL = 4
TASKS = [(ph, i) for ph in PHASES for i in range(N_REAL)]
SEED_NOISE_BASE = 40000
N_THREADS = int(os.environ.get('SLURM_NTASKS', 4))
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
TAG_PARAMS_OUT = '_biasshame_noisebest_p0_n1'
# the 2026-06-03 pick (best of the 1000-mock LH, library units), kept for comparison
A_OLD_LIB = np.array([2.346482608623328, -1.3433619456637138, -0.226130277108882, 0.5266325796844749, 1.6813773681515105])


def make_bspec(Pk=None):
    # production binning (compute_statistics.compute_bispectrum); Pk=None -> P_fid = 1 (production)
    import PolyBin3D as pb
    base = pb.PolyBin3D([cnp.L] * 3, cnp.NG, Pk=Pk, backend='fftw', nthreads=N_THREADS, sightline='global')
    k_edges = np.linspace(0.01, 0.4, 13)
    return pb.BSpec(base, k_edges, applySinv=None, mask=None, lmax=1,
                    k_bins_squeeze=k_edges.copy(), include_partial_triangles=False)


def setup_bk():
    bspec = make_bspec()
    return bspec.get_ks(), lambda f: bspec.Bk_ideal(f.astype(np.float64), discreteness_correction=True)['b0']


def gaussian_var_B(k, P):
    # Gaussian B variance per triangle: diag of the inverse ideal Fisher matrix with P_fid = P
    kk = np.concatenate([[0.], k, [10.]])
    PP = np.concatenate([[P[0]], P, [P[-1]]])    # constant extrapolation outside the measured bins
    F = make_bspec(Pk=[kk, PP]).compute_fisher_ideal(discreteness_correction=True)
    return np.diag(np.linalg.inv(F))


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
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    np.save(DIR_OUT / f'task{task}.npy', out)


# ---------------------------------------------------------------- targets
def targets():
    # SHAMe P, P_gm (vs O_1 - 1), B for all densities (pair mean), plus their Gaussian variances
    k123, Bk = setup_bk()
    out = {}
    for tag in TAGS_SHAME:
        per_phase = []
        for phase in PHASES:
            ops = cnp.load_ops(phase)
            dm_k = cnp.ft(ops[0] - 1 - (ops[0] - 1).mean())
            tr = np.load(cnp.msm.DIR_DATA_SHAME / f'data{tag}' / f'tracer_field_phase{phase}.npy').astype(np.float64) * cnp.NG**3
            tr = tr - tr.mean()
            tr_k = cnp.ft(tr)
            per_phase.append(dict(pk=cnp.pk(tr_k, tr_k), pgm=cnp.pk(tr_k, dm_k), bk=Bk(tr)))
        sh = {st: np.mean([d[st] for d in per_phase], axis=0) for st in per_phase[0]}
        sh['var'] = dict(pk=2 * sh['pk']**2 / cnp.nmode, bk=gaussian_var_B(cnp.k_pk, sh['pk']),
                         pgm=(sh['pk']**2 + sh['pgm']**2) / cnp.nmode)   # pgm: rough Gaussian (weight 0)
        out[tag] = sh
        print(f'{tag}: targets done', flush=True)
    out['k_pk'], out['k123'] = cnp.k_pk, k123
    np.save(DIR_OUT / 'targets.npy', out)


# ---------------------------------------------------------------- fit
def load_tasks():
    return {i: np.load(DIR_OUT / f'task{i}.npy', allow_pickle=True).item() for i in range(len(TASKS))}


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


def bias_settings(tag):
    # fixed biases: bias_dict_shame as in the code (b2, bs2 = 2 x raw), and raw / 2 (Stuecker+24 -> our basis)
    import utils_model
    b_code = np.array([utils_model.bias_dict_shame[tag][n] for n in BIAS_NAMES])
    b_half = b_code * np.array([1, 0.25, 0.25, 1])
    return {'fixed b (code, x2)': b_code, 'fixed b (raw/2)': b_half, 'free b': None}


def fit():
    from scipy.optimize import minimize
    import generate_params as genp

    t = load_tasks()
    tg = np.load(DIR_OUT / 'targets.npy', allow_pickle=True).item()
    stats = make_stats(t[0]['expo'])
    C = coeffs(t, range(len(TASKS)))                 # expectation over eps (8 = 2 phases x 4 realizations)
    k, k123 = tg['k_pk'], tg['k123']
    masks = dict(pk=np.ones(len(k), bool), bk=k123.max(axis=0) < KMAX_B, pgm=np.ones(len(k), bool))
    wkey = dict(pk='pk', bk='bispec', pgm='pgm')
    # A_homog >= 0 removes the exact global-sign duplicate (t(A, eps) = t(-A, -eps)); others unbounded,
    # so the fits show where the data wants the amplitudes regardless of the current priors
    bnds_A = [(0, None)] + [(None, None)] * 4
    rng = np.random.default_rng(0)
    starts_A = [A_OLD_LIB * V_SCALE] + \
               [rng.uniform(-3, 3, 5) * np.array([0.5, 1, 1, 1, 3]) + np.array([1.5, 0, 0, 0, 0]) for _ in range(15)]

    out = dict(k_pk=k, k123=k123, masks=masks, V_scale=V_SCALE, bias_names=BIAS_NAMES, noise_names=NOISE_NAMES,
               bounds_A={n: tuple(genp.BOUNDS['Anoisemult'][n]) for n in NOISE_NAMES}, targets=tg, fits={})
    for tag in TAGS_SHAME:
        sh, var = tg[tag], tg[tag]['var']

        def chi2_red(x):
            s = stats(x, C)
            return {p: np.mean((s[p][m] - sh[p][m])**2 / var[p][m]) for p, m in masks.items()}

        def loss(x):
            c = chi2_red(x)
            return sum(PROBE_WEIGHTS[wkey[p]] * c[p] for p in c)

        for setting, b in bias_settings(tag).items():
            if b is None:
                b0 = bias_settings(tag)['fixed b (raw/2)']
                f = lambda z: loss(z)
                starts, bnds = [np.concatenate([b0, a]) for a in starts_A], [(None, None)] * 4 + bnds_A
            else:
                f = lambda z, b=b: loss(np.concatenate([b, z]))
                starts, bnds = starts_A, bnds_A
            best = min((minimize(f, z0, method='L-BFGS-B', bounds=bnds) for z0 in starts), key=lambda r: r.fun)
            x = best.x if b is None else np.concatenate([b, best.x])
            c = chi2_red(x)
            out['fits'][(tag, setting)] = dict(b=x[:4], A_lib=x[4:] / V_SCALE, chi2_red=c, loss=loss(x), stats=stats(x, C))
            print(f'{tag} {setting:18s} loss={loss(x):6.2f} chi2r P={c["pk"]:5.2f} B={c["bk"]:5.2f} Pgm={c["pgm"]:6.2f}  '
                  f'b={np.round(x[:4], 3)}  A_lib={np.round(x[4:] / V_SCALE, 3)}', flush=True)

    # the test-set selection: nbar = 2.2e-4 with the biases the test set uses (code convention)
    sel = out['fits'][(TAG_MOCK, 'fixed b (code, x2)')]
    out_of_prior = [n for n, a in zip(NOISE_NAMES, sel['A_lib']) if not (out['bounds_A'][n][0] <= a <= out['bounds_A'][n][1])]
    if out_of_prior:
        print('WARNING: selected amplitudes outside the current priors:', out_of_prior)
    out['selected'] = dict(zip(BIAS_NAMES + NOISE_NAMES, map(float, np.concatenate([sel['b'], sel['A_lib']]))))
    np.save(DIR_OUT / 'fit.npy', out)
    genp.write_authored_params_fixed(TAG_PARAMS_OUT, out['selected'], param_names=BIAS_NAMES + NOISE_NAMES)
    print(f'Saved {DIR_OUT / "fit.npy"}')


if __name__ == '__main__':
    {'emulate': lambda: emulate(int(sys.argv[2])), 'targets': targets, 'fit': fit}[sys.argv[1]]()
