"""compute_shame_matched_noisepoly.py

Polynomial "emulator" of P(k) and B(k1,k2,k3) on the phase-matched muchisimocks fields as a
function of the noise amplitudes, for the noise-model test in
notebooks/2026-11-12_final_checks.ipynb (section 2.4).

The tracer is linear in the noise amplitudes,
    t = c_0 + sum_j A_j c_j,
    c_0 = deterministic tracer  (O_1 - 1) + sum_i b_i O_i  (field-level-fit biases, fixed)
    c_1..c_5 = eps * O_i           (i = 1, delta, delta^2, s^2, lapl; the 5-param multiplicative model)
    c_6 = (eps^2 - 1) * O_1        (second-order homogeneous term)
    c_7 = (eps^2 - 1) * O_lapl     (second-order lapl term)
so P(t) is exactly quadratic in A (stored as the 8x8 cross-spectrum matrix) and B(t) is
exactly cubic (fit by least squares over random A, checked on held-out points).
Fields are normalized as in the production pipeline (O_1 = 1 + delta_m), so the A_j here are
the same amplitudes as in utils_model.get_tracer_field. eps^2 - 1 (not eps^2) removes the mean,
which would otherwise just rescale the O_1 term.

Output: data/data_shame/matched_seed-406168/noise_poly.npy
"""
import sys
import time
import itertools
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

import paths
import compute_statistics as cs
import make_shame_matched_mock as msm

SEED = -406168
L, NG, NG_DEP = msm.BOX_SIZE, msm.N_GRID_TARGET, msm.N_GRID
TAG_MOCK = '_nbar0.00022'
PHASES = ['0', 'pi']
N_REAL = 2
SEED_NOISE_BASE = 40000          # same eps as compute_shame_matched_noisebest.py, i = 0, 1
N_THREADS = 16
N_DESIGN = 160                   # > 120 cubic monomials in 7 variables
N_TEST = 8
COMP_NAMES = ['det', 'eps*O1', 'eps*delta', 'eps*delta2', 'eps*s2', 'eps*lapl',
              '(eps2-1)*O1', '(eps2-1)*lapl']
FN_OUT = msm.DIR_DATA_SHAME / f'matched_seed{SEED}' / 'noise_poly.npy'

# ---- numpy P(k) estimator, dk = k_f (identical to the notebook's pk_sh) ----
kf = 2 * np.pi / L
kx = np.fft.fftfreq(NG, d=1. / NG) * kf
kz = np.fft.rfftfreq(NG, d=1. / NG) * kf
KMAG = np.sqrt(kx[:, None, None]**2 + kx[None, :, None]**2 + kz[None, None, :]**2)
edges = np.arange(0.5, NG // 2) * kf
k_pk = 0.5 * (edges[1:] + edges[:-1])
idx = np.digitize(KMAG.ravel(), edges) - 1
ok = (idx >= 0) & (idx < len(k_pk))
w = np.full(KMAG.shape, 2.0); w[..., 0] = 1; w[..., -1] = 1
w = w.ravel()[ok]
nmode = np.bincount(idx[ok], w, len(k_pk))


def ft(f):
    return np.fft.rfftn(f - f.mean()) * (L / NG)**3


def pk(fa, fb):
    p = (fa * np.conj(fb)).real.ravel()[ok] * w
    return np.bincount(idx[ok], p, len(k_pk)) / nmode / L**3


def monomials(nvar, deg=3):
    out = []
    for d in range(deg + 1):
        for combo in itertools.combinations_with_replacement(range(nvar), d):
            e = np.zeros(nvar, int)
            for c in combo:
                e[c] += 1
            out.append(e)
    return np.array(out)


def design(U, expo):
    return np.prod(U[:, None, :] ** expo[None, :, :], axis=-1)


KMAX_BFIT = 0.2


def load_ops(phase):
    fn = paths.mocks_lib_ood_dir(msm.SUBDIR_OOD) / f'phase{phase}_seed{SEED}' / f'bias_fields_eul_deconvolved_{SEED}.npy'
    ops = np.load(fn).astype(np.float64)
    return ops / ops[0].mean()                    # O_1 -> 1 + delta_m (same factor for all operators)


def load_shame_tracer(phase):
    tr = np.load(msm.DIR_DATA_SHAME / f'data{TAG_MOCK}' / f'tracer_field_phase{phase}.npy').astype(np.float64) * NG**3
    return tr - tr.mean()


def fit_biases():
    # field-level least squares over modes k <= KMAX_BFIT, averaged over the two phases
    # (same as fit_bias_fieldlevel in the notebook)
    m = (KMAG <= KMAX_BFIT) & (KMAG > 0)
    bs = []
    for phase in PHASES:
        ops_k = [ft(o) for o in load_ops(phase)]
        y = (ft(load_shame_tracer(phase)) - ops_k[0])[m]
        A = np.stack([o[m] for o in ops_k[1:]], axis=1)
        bb, *_ = np.linalg.lstsq(np.concatenate([A.real, A.imag]), np.concatenate([y.real, y.imag]), rcond=None)
        bs.append(bb)
    return np.mean(bs, axis=0)


def main():
    b = fit_biases()
    print('biases (field-level fit, phase mean):', b, flush=True)

    import PolyBin3D as pb
    base = cs.setup_bispec(L, NG, n_threads=N_THREADS)
    k_edges = np.linspace(0.01, 0.4, 13)
    bspec = pb.BSpec(base, k_edges, applySinv=None, mask=None, lmax=1,
                     k_bins_squeeze=k_edges.copy(), include_partial_triangles=False)
    k123 = bspec.get_ks()
    Bk = lambda f: bspec.Bk_ideal(f.astype(np.float64), discreteness_correction=True)['b0']

    ncomp = len(COMP_NAMES)
    nvar = ncomp - 1
    expo = monomials(nvar)
    rng_design = np.random.default_rng(123)
    out = dict(comp_names=COMP_NAMES, k_pk=k_pk, k123=k123, weight=k123.prod(axis=0), b=b,
               expo=expo, n_real=N_REAL, tag_mock=TAG_MOCK)

    for phase in PHASES:
        ops = load_ops(phase)
        det = (ops[0] - 1) + sum(bi * o for bi, o in zip(b, ops[1:]))

        tr = load_shame_tracer(phase)
        tr_k = ft(tr)
        out[('shame', phase)] = dict(pk=pk(tr_k, tr_k), bk=Bk(tr))
        print(f'phase {phase}: SHAMe B computed', flush=True)

        for i in range(N_REAL):
            eps = np.random.default_rng(SEED_NOISE_BASE + i).standard_normal(ops[0].shape)
            comps = [det] + [eps * o for o in ops] + [(eps**2 - 1) * ops[0], (eps**2 - 1) * ops[4]]
            comps = [c - c.mean() for c in comps]
            comps_k = [ft(c) for c in comps]
            Pmat = np.array([[pk(comps_k[a], comps_k[c]) for c in range(ncomp)] for a in range(ncomp)])
            # scale each amplitude so its component has ~half the power of det (conditioning)
            s = np.array([0.5 * comps[0].std() / comps[j].std() for j in range(1, ncomp)])

            def tracer(u):
                return comps[0] + sum(uj * sj * c for uj, sj, c in zip(u, s, comps[1:]))

            U = rng_design.standard_normal((N_DESIGN + N_TEST, nvar))
            t0 = time.time()
            Bs = np.array([Bk(tracer(u)) for u in U])
            X = design(U[:N_DESIGN], expo)
            coef, *_ = np.linalg.lstsq(X, Bs[:N_DESIGN], rcond=None)
            pred = design(U[N_DESIGN:], expo) @ coef
            err = np.abs(pred / Bs[N_DESIGN:] - 1).max()
            # exactness check of P quadratic form on a test point
            u = U[-1]; a = np.concatenate([[1.], u * s])
            p_q = np.einsum('a,c,ack->k', a, a, Pmat)
            tk = ft(tracer(u)); p_d = pk(tk, tk)
            print(f'phase={phase} real={i}: {len(U)} B evals in {time.time() - t0:.0f} s; '
                  f'held-out max|B_poly/B - 1| = {err:.2e}; P quad check = {np.abs(p_q / p_d - 1).max():.2e}', flush=True)
            out[(phase, i)] = dict(Pmat=Pmat, Bcoef=coef, scale=s, B_holdout_err=err)
        np.save(FN_OUT, out)
    print(f'Saved {FN_OUT}')


if __name__ == '__main__':
    main()
