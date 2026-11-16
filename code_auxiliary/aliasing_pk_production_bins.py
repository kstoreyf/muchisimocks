"""P(k) of the deposit-aliasing test fields, measured exactly as in production.

code/test_deposit_aliasing.py saves the k-cut, deconvolved operator fields for each deposit mesh
(bias_fields_<cfg>.npy) but measures P(k) with its own linear-bin estimator. This script builds the
tracer and measures P(k) with production's utils_model.get_tracer_field + compute_statistics.compute_pk
(bacco, 32 log bins 0.01-0.4 h/Mpc, (Mpc/h)^3), so the numbers are on the analysis binning.

Output: <DIR_DATA>/aliasing_test/LH0/pk_production_bins.npz with k, pk_<field>_<cfg>.
    sbatch jobs/job_aliasing_pk_production_bins.s      (from code_auxiliary/)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import compute_statistics  # noqa: E402
import paths  # noqa: E402
import test_deposit_aliasing as tda  # noqa: E402
import utils_model  # noqa: E402

CFGS = ["ng256", "ng512", "ng1024"]
N_GRID_ORIG = 512  # particle grid; production normalises the tracer by this (compute_statistics)
TAG_BIAS = "_nbar0.00022"


def main():
    dir_alias = paths.DIR_DATA / "aliasing_test" / "LH0"
    param_dict = tda.load_param_dict(tda.DIR_RERUN, 0)
    cosmo = utils_model.get_cosmo(param_dict, a_scale=1.0, sim_name="quijote")
    bias = utils_model.bias_dict_shame[TAG_BIAS]
    bias_vector = [bias[n] for n in utils_model.biasparam_names_ordered]

    out = {}
    for c in CFGS:
        fields = np.load(dir_alias / f"bias_fields_{c}.npy")
        grids = {"tracer": utils_model.get_tracer_field(fields, bias_vector, N_GRID_ORIG)}
        grids.update({op: fields[i] / N_GRID_ORIG**3 for i, op in enumerate(tda.OP_NAMES)})
        for name, g in grids.items():
            pk = compute_statistics.compute_pk(np.ascontiguousarray(g, dtype=np.float32), cosmo, 1000.0,
                                               n_threads=tda.N_THREADS)
            out["k"] = np.asarray(pk["k"])
            out[f"pk_{name}_{c}"] = np.asarray(pk["pk"])
        print(c, "tracer P(k) at k~0.1:", out[f"pk_tracer_{c}"][np.argmin(np.abs(out["k"] - 0.1))], flush=True)
    out["bias_tag"] = TAG_BIAS
    np.savez(dir_alias / "pk_production_bins.npz", **out)
    print("saved", dir_alias / "pk_production_bins.npz")


if __name__ == "__main__":
    main()
