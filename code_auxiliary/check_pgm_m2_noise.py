"""check_pgm_m2_noise.py

Rough check: can second-order (m=2) noise terms help model P_gm of the phase-matched SHAMe mock?

Matter field = O_1 - 1 (the advected "1" operator, as in v2 P_gm). Tracer model
    t = (O_1 - 1) + sum_i b_i O_i + sum_j A_j c_j
P_gm is linear in A, so each candidate term enters only through P(c_j, delta_m):
  - eps*O_i, (eps^2-1)*O_i: zero-mean noise factor -> expectation 0 (pure scatter); we measure
    its size vs the deterministic P_gm.
  - uncentered eps^2*O_i = O_i + (eps^2-1)*O_i: deterministic part just shifts b_i (degenerate)
    except for O_1, whose coefficient is otherwise fixed to 1 -> test a free O_1 amplitude.
Fits: P_gm-only least squares over k in [KMIN, KMAX] with biases free, with/without a_1.

Run (benv): python check_pgm_m2_noise.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

import compute_shame_matched_noisepoly as cnp   # numpy P(k) estimator + field loaders

PHASES = cnp.PHASES
KMIN, KMAX = 0.03, 0.35
N_REAL = 2
OPS = ['O1', 'delta', 'delta2', 's2', 'lapl']


def main():
    k = cnp.k_pk
    m = (k >= KMIN) & (k <= KMAX)
    b_field = cnp.fit_biases()
    print('field-level biases (b1, b2, bs2, bl):', np.round(b_field, 4))

    X_list, y_list, noise_rel = [], [], {}
    for phase in PHASES:
        ops = cnp.load_ops(phase)
        dm_k = cnp.ft(ops[0] - 1)
        ops_k = [cnp.ft(o) for o in ops]
        Pi = np.array([cnp.pk(ok_, dm_k) for ok_ in ops_k])   # P(O_i, delta_m), i = 0..4
        tr_k = cnp.ft(cnp.load_shame_tracer(phase))
        y_list.append(cnp.pk(tr_k, dm_k))
        X_list.append(Pi)
        P_det = Pi[0] + b_field @ Pi[1:]
        for i in range(N_REAL):
            eps = np.random.default_rng(cnp.SEED_NOISE_BASE + i).standard_normal(ops[0].shape)
            for name, fac in [('eps', eps), ('eps2-1', eps**2 - 1)]:
                for j, o in enumerate(ops):
                    c = fac * o
                    pc = cnp.pk(cnp.ft(c - c.mean()), dm_k)
                    # per unit "natural" amplitude: scale c to have the std of the det field
                    det = (ops[0] - 1) + sum(bb * oo for bb, oo in zip(b_field, ops[1:]))
                    s = det.std() / c.std()
                    noise_rel.setdefault(f'{name}*{OPS[j]}', []).append(np.abs(s * pc[m] / P_det[m]))

    print(f'\n|P(c_j, delta_m)| / P_gm,det for c_j normalized to the det field std, k in [{KMIN},{KMAX}]')
    print('(expectation zero; this is scatter from one 1 (Gpc/h)^3 box, per phase x realization)')
    for name, v in noise_rel.items():
        v = np.concatenate(v)
        print(f'  {name:14s} median {np.median(v):.1e}   max {v.max():.1e}')

    y = np.mean(y_list, axis=0)
    X = np.mean(X_list, axis=0)   # (5, nk)
    w = 1 / y[m]                  # fractional residuals

    def fit(free_a1):
        # model: a1 * P(O1,dm) + sum b_i P(O_i,dm); a1 = 1 unless free
        if free_a1:
            A = X[:, m].T * w[:, None]
            coef, *_ = np.linalg.lstsq(A, y[m] * w, rcond=None)
        else:
            A = X[1:, m].T * w[:, None]
            c, *_ = np.linalg.lstsq(A, (y[m] - X[0, m]) * w, rcond=None)
            coef = np.concatenate([[1.], c])
        return coef, coef @ X / y - 1

    rows = [('field-level b, a1=1', np.concatenate([[1.], b_field]), None)]
    for free in (False, True):
        coef, r = fit(free)
        rows.append((f'P_gm fit, b free{", a1 free" if free else ", a1=1"}', coef, r))
    print(f'\nFits to SHAMe P_gm (phase mean), fractional residuals over k in [{KMIN},{KMAX}]:')
    for label, coef, r in rows:
        if r is None:
            r = coef @ X / y - 1
        print(f'  {label:28s} a1={coef[0]:.4f} b={np.round(coef[1:], 3)}  '
              f'median|r|={np.median(np.abs(r[m])):.2e}  max|r|={np.abs(r[m]).max():.2e}')
    kk = [0.05, 0.1, 0.2, 0.3]
    print('\n  residual at k =', kk)
    for label, coef, r in rows:
        if r is None:
            r = coef @ X / y - 1
        print(f'  {label:28s}', np.round(np.interp(kk, k, r), 4))


if __name__ == '__main__':
    main()
