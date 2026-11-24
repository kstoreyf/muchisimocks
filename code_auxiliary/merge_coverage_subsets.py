"""merge_coverage_subsets.py

Merge coverage-test outputs run on complementary row subsets (e.g. inner500 + outer500) into the full-set files,
so everything that reads the full coverage test (plotter_figs, choose_best, notebooks) works unchanged and nothing
is rerun. Agreed 2026-11-23: full 1000-row coverage = inner500 (run first) + outer500, merged.

For each model dir and test tag (the full-set tag, without the subset suffix), and each output kind:
  samples_test{tag}_{sub}_pred.npy        (n_draws, n_obs_sub, n_params)  -> samples_test{tag}_pred.npy
  samples_test{tag}_{sub}_acceptance.npy  (n_obs_sub,)                    -> samples_test{tag}_acceptance.npy
rows are placed at data/params/idxs_obs{tag_params_test}_{sub}.npy. The subsets must be disjoint and cover every
row; draws per row must agree. A merge record is written next to the output
(samples_test{tag}_pred.merged.json). Existing full-set files are never overwritten.

Run with env_v2.sh sourced (loads ~0.5 GB per subset; use srun/sbatch, not the login node):
  python merge_coverage_subsets.py --tag-test <full test tag> --subsets inner500 outer500 <model dir> [...]
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import paths  # noqa: E402

DIR_PARAMS = Path(paths.PROJECT_ROOT) / "data" / "params"
DIR_SBI = Path(paths.DIR_RESULTS) / "results_sbi"
OBS_AXIS = {"pred": 1, "acceptance": 0}


def tag_params_test(tag_test):
    """'_muchisimocks_<stats+masks>_coveragev2_p5_n1000_biascov..._noisefly' -> '_coveragev2_p5_n1000'."""
    i = tag_test.index("_coverage")
    j = tag_test.index("_bias", i)
    return tag_test[i:j]


def merge_dir(d, tag_test, subsets):
    d = Path(d)
    if not d.is_absolute():
        d = DIR_SBI / d
    idxs = {s: np.load(DIR_PARAMS / f"idxs_obs{tag_params_test(tag_test)}_{s}.npy") for s in subsets}
    allidx = np.concatenate(list(idxs.values()))
    n_obs = len(allidx)
    assert len(np.unique(allidx)) == n_obs and allidx.min() == 0 and allidx.max() == n_obs - 1, \
        f"subsets {subsets} are not a disjoint cover of 0..{n_obs - 1}"
    out_pred = d / f"samples_test{tag_test}_pred.npy"
    if out_pred.exists():
        print(f"skip {d.name}: {out_pred.name} exists")
        return False
    src = {k: {s: d / f"samples_test{tag_test}_{s}_{k}.npy" for s in subsets} for k in OBS_AXIS}
    missing = [str(f.name) for f in src["pred"].values() if not f.is_file()]
    if missing:
        print(f"skip {d.name}: missing {missing}")
        return False
    for kind, files in src.items():
        if not all(f.is_file() for f in files.values()):
            print(f"  {kind}: not all subsets have it, not merged")
            continue
        parts = {s: np.load(f) for s, f in files.items()}
        ax = OBS_AXIS[kind]
        for s, a in parts.items():
            assert a.shape[ax] == len(idxs[s]), (kind, s, a.shape, len(idxs[s]))
        ref = next(iter(parts.values()))
        shape = list(ref.shape); shape[ax] = n_obs
        if kind == "pred":
            assert len({a.shape[0] for a in parts.values()}) == 1, "draws per row differ between subsets"
        full = np.full(shape, np.nan, dtype=ref.dtype)
        for s, a in parts.items():
            sl = [slice(None)] * full.ndim; sl[ax] = idxs[s]
            full[tuple(sl)] = a
        out = d / f"samples_test{tag_test}_{kind}.npy"
        tmp = out.with_name(out.name + ".tmp.npy")
        np.save(tmp, full)
        os.replace(tmp, out)
        print(f"  wrote {out.name} {tuple(full.shape)}")
    rec = {"merged_at": datetime.datetime.now().isoformat(timespec="seconds"), "subsets": list(subsets),
           "sources": {k: {s: f.name for s, f in v.items()} for k, v in src.items()},
           "script": "code_auxiliary/merge_coverage_subsets.py"}
    (d / f"samples_test{tag_test}_pred.merged.json").write_text(json.dumps(rec, indent=1) + "\n")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-test", required=True, help="full-set test tag, e.g. _muchisimocks_pk_..._noisefly")
    ap.add_argument("--subsets", nargs="+", default=["inner500", "outer500"])
    ap.add_argument("dirs", nargs="+", help="model dirs (sbi_muchisimocks_...), absolute or under results_sbi")
    a = ap.parse_args()
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    for d in a.dirs:
        print(Path(d).name)
        merge_dir(d, a.tag_test, a.subsets)


if __name__ == "__main__":
    main()
