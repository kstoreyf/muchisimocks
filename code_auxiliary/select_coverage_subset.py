"""select_coverage_subset.py

Pick the coverage-set rows farthest from the training-prior edges, then a random subset of those,
for quick coverage tests.

Distance to the edge: each varied parameter is mapped to u in [0, 1] over its training prior box
(generate_params.BOUNDS; physical bias/noise box, i.e. the box the training rows were drawn from).
The edge distance of a row is min_j min(u_j, 1 - u_j) over the three key parameters
(PARAMS_EDGE: omega_cold, sigma8_cold, b1), where prior truncation would bias the headline posteriors.
(Over all 14 varied params the inner 500 would only be >= 2.4% of each width from the edges.) Keep the N_INNER rows with the
largest edge distance, then draw N_RAND of them.

Output: data/params/idxs_obs_coveragev2_p5_n1000_inner{N_INNER}_rand{N_RAND}.npy (sorted row indices)
  python select_coverage_subset.py      (seconds; fine on a login node)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
import generate_params as genp  # noqa: E402

TAG_PARAMS, TAG_BIASPARAMS = "_coveragev2_p5_n1000", "_biasnoisecoveragev2_p9_n1000"
N_INNER, N_RAND, SEED = 500, 100, 42
PARAMS_EDGE = ["omega_cold", "sigma8_cold", "b1"]
DIR_PARAMS = REPO / "data" / "params"

cosmo = pd.read_csv(DIR_PARAMS / f"params_lh{TAG_PARAMS}.txt", index_col=0)
bias = pd.read_csv(DIR_PARAMS / f"params_lh{TAG_BIASPARAMS}.txt", index_col=0)
assert len(cosmo) == len(bias) and (cosmo.index == bias.index).all()
bounds = {**genp.BOUNDS["cosmo"], **genp.BOUNDS["bias"], **genp.BOUNDS["Anoisemult"]}
df = pd.concat([cosmo, bias], axis=1)
u_all = np.stack([(df[p] - bounds[p][0]) / (bounds[p][1] - bounds[p][0]) for p in df.columns], axis=1)
assert (u_all >= 0).all() and (u_all <= 1).all(), "coverage row outside the training prior"
u = u_all[:, [list(df.columns).index(p) for p in PARAMS_EDGE]]
d_edge = np.minimum(u, 1 - u).min(axis=1)

order = np.argsort(-d_edge, kind="stable")
idxs_inner = np.sort(order[:N_INNER])
idxs = np.sort(np.random.default_rng(SEED).choice(idxs_inner, N_RAND, replace=False))
print(f"edge distance over {PARAMS_EDGE}")
print(f"inner {N_INNER}: edge distance >= {d_edge[order[N_INNER - 1]]:.4f} of each prior width "
      f"(all 1000: median {np.median(d_edge):.4f})")

for fn_tag, arr in ((f"inner{N_INNER}", idxs_inner), (f"inner{N_INNER}_rand{N_RAND}", idxs)):
    fn = DIR_PARAMS / f"idxs_obs{TAG_PARAMS}_{fn_tag}.npy"
    np.save(fn, arr)
    print(f"saved {fn} ({len(arr)} rows)")
