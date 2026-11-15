"""check_bispec_gaussian_var.py

Check the normalization of the Gaussian B variance used in select_shame_noisebest.py:
for unit-variance white noise on the matched grid (P = V_cell exactly), the empirical
per-triangle variance of the production B estimator over N sims should equal
diag(F^-1) from PolyBin's ideal Fisher matrix with P_fid = V_cell (ratio ~ 1 +- sqrt(2/(N-1))).
Job: code_auxiliary/jobs/job_check_bispec_gaussian_var.s
"""
import numpy as np
import select_shame_noisebest as s
import compute_shame_matched_noisepoly as cnp

N = 40
k123, Bk = s.setup_bk()
Vc = (cnp.L / cnp.NG)**3
rng = np.random.default_rng(7)
Bs = np.array([Bk(rng.standard_normal((cnp.NG,) * 3)) for _ in range(N)])
var_g = s.gaussian_var_B(cnp.k_pk, np.full(len(cnp.k_pk), Vc))
r = Bs.var(axis=0, ddof=1) / var_g
m = k123.max(axis=0) < s.KMAX_B
print(f'empirical / Fisher variance, {N} white-noise sims (expected 1 +- {np.sqrt(2 / (N - 1)):.2f} per triangle):')
print(f'  all triangles: median {np.median(r):.3f}, 16-84% {np.percentile(r, 16):.3f}-{np.percentile(r, 84):.3f}')
print(f'  k_i < {s.KMAX_B}:  median {np.median(r[m]):.3f}, 16-84% {np.percentile(r[m], 16):.3f}-{np.percentile(r[m], 84):.3f}')
