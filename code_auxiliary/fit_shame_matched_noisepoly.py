"""fit_shame_matched_noisepoly.py

Fit noise amplitudes on the phase-matched mock using the exact P (quadratic) / B (cubic)
polynomials from compute_shame_matched_noisepoly.py, for several sets of noise terms.
Results are plotted in notebooks/2026-11-12_final_checks.ipynb section 2.4.

Fast: the polynomial coefficients are converted once to amplitude space and averaged over
phases and noise realizations, so each loss evaluation is a single small matrix product.
Runs in about a minute on a login node:  python fit_shame_matched_noisepoly.py

Output: data/data_shame/matched_seed-406168/noise_poly_fits.npy
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')   # tiny matmuls: multithreaded BLAS is ~35x slower on a busy node
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

import make_shame_matched_mock as msm

SEED = -406168
DIR = msm.DIR_DATA_SHAME / f'matched_seed{SEED}'
FN_IN, FN_OUT = DIR / 'noise_poly.npy', DIR / 'noise_poly_fits.npy'
PHASES = ['0', 'pi']
KMIN_P, KMAX_P, KMAX_B = 0.03, 0.35, 0.25
N_START = 10

# indices into the 7 amplitudes: eps*O_i (0-4: 1, delta, delta2, s2, lapl), (eps^2-1)*O_1 (5), (eps^2-1)*O_lapl (6)
NOISE_MODELS = {
    'none':        [],
    'eps':         [0, 1, 2, 3, 4],
    'eps+eps2_O1': [0, 1, 2, 3, 4, 5],
    'eps+eps2_lapl': [0, 1, 2, 3, 4, 6],
    'm2p3':        [0, 5, 6],
}
LABELS = {
    'none': 'none (deterministic only)',
    'eps': r'$\epsilon$ terms (5-param, paper)',
    'eps+eps2_O1': r'$\epsilon$ + $(\epsilon^2-1)\mathcal{O}_1$',
    'eps+eps2_lapl': r'$\epsilon$ + $(\epsilon^2-1)\mathcal{O}_{\nabla^2\delta}$',
    'm2p3': r'm2p3 basis ($\epsilon\mathcal{O}_1$, $\epsilon^2\mathcal{O}_1$, $\epsilon^2\mathcal{O}_{\nabla^2\delta}$)',
}

poly = np.load(FN_IN, allow_pickle=True).item()
expo = poly['expo']
deg = expo.sum(axis=1)
k_p, k123 = poly['k_pk'], poly['k123']
P_sh = np.mean([poly[('shame', ph)]['pk'] for ph in PHASES], axis=0)
B_sh = np.mean([poly[('shame', ph)]['bk'] for ph in PHASES], axis=0)
mP = (k_p >= KMIN_P) & (k_p <= KMAX_P)
mB = k123.max(axis=0) < KMAX_B
mB_hi = k123.max(axis=0) > 0.3


def coeffs(reals):
    # amplitude-space coefficients averaged over (phase, realization)
    Pmat = np.mean([poly[r]['Pmat'] for r in reals], axis=0)
    Bc = np.mean([poly[r]['Bcoef'] / np.prod(poly[r]['scale'][None, :] ** expo, axis=1)[:, None]
                  for r in reals], axis=0)
    return Pmat, Bc


def stats(A, C):
    Pmat, Bc = C
    a = np.concatenate([[1.], A])
    P = np.einsum('a,c,ack->k', a, a, Pmat)
    B = np.prod(A[None, :] ** expo, axis=1) @ Bc
    return P, B


def loss(A, C):
    P, B = stats(A, C)
    return np.mean((P[mP] / P_sh[mP] - 1)**2) + np.mean((B[mB] / B_sh[mB] - 1)**2)


def fit(idx, C, scale, A_init=None, seed=0):
    # random starts, plus A_init (e.g. the nested 5-param solution) so nested models can't do worse
    if not idx:
        return np.zeros(7)
    rng = np.random.default_rng(seed)

    def f(x):
        A = np.zeros(7); A[idx] = x
        return loss(A, C)
    x0s = [rng.standard_normal(len(idx)) * scale[idx] for _ in range(N_START)]
    if A_init is not None:
        x0s.append(A_init[idx])
    best = min((minimize(f, x0, method='L-BFGS-B') for x0 in x0s), key=lambda r: r.fun)
    A = np.zeros(7); A[idx] = best.x
    return A


def B_by_order(A, C):
    _, Bc = C
    mono = np.prod(A[None, :] ** expo, axis=1)
    return np.array([(mono * (deg == d)) @ Bc for d in range(4)])


def main():
    t0 = time.time()
    reals = [(ph, i) for ph in PHASES for i in range(poly['n_real'])]
    C_all = coeffs(reals)
    C_fit, C_test = coeffs([(ph, 0) for ph in PHASES]), coeffs([(ph, 1) for ph in PHASES])
    scale = np.mean([poly[r]['scale'] for r in reals], axis=0)

    out = dict(k_pk=k_p, k123=k123, weight=poly['weight'], b=poly['b'], comp_names=poly['comp_names'],
               P_sh=P_sh, B_sh=B_sh, mP=mP, mB=mB, mB_hi=mB_hi, labels=LABELS, models={})
    for name, idx in NOISE_MODELS.items():
        nested = name != 'eps' and set(NOISE_MODELS['eps']) <= set(idx)
        A = fit(idx, C_all, scale, A_init=out['models']['eps']['A'] if nested else None)
        P, B = stats(A, C_all)
        rP, rB = P / P_sh - 1, B / B_sh - 1
        A_fit = fit(idx, C_fit, scale, A_init=A)
        out['models'][name] = dict(
            A=A, P=P, B=B, B_order=B_by_order(A, C_all), loss=loss(A, C_all),
            P_med=np.median(np.abs(rP[mP])), P_max=np.abs(rP[mP]).max(),
            B_med=np.median(np.abs(rB[mB])), B_max=np.abs(rB[mB]).max(),
            B_hi_med=np.median(rB[mB_hi]), B_hi_90=np.percentile(np.abs(rB[mB_hi]), 90),
            loss_holdout=(loss(A_fit, C_fit), loss(A_fit, C_test)))
        print(f'{name:14s} loss={out["models"][name]["loss"]:.1e}  A={np.round(A, 3)}  ({time.time() - t0:.0f} s)', flush=True)
    np.save(FN_OUT, out)
    print(f'Saved {FN_OUT}')


if __name__ == '__main__':
    main()
