"""Precompute the scale-cut masks for every K_CUTS label and statistic (once per data tree).

A cut keeps every bin whose top edge is <= the label (see data_loader "Scale cuts"); masks go
to <DIR_DATA>/masks/mask_<stat><tag>.txt and are only *loaded* by the pipeline.

Run from code/ with the data tree's env sourced, e.g.
    source ../code_auxiliary/env_v2.sh && python ../code_auxiliary/make_masks.py
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import data_loader  # noqa: E402
import paths  # noqa: E402

# statistic -> (statistics dir prefix under DIR_DATA, file prefix)
STAT_FILES = {"pk": "pks_mlib", "pgm": "pgms_mlib", "bispec": "bispecs_mlib"}


def sample_k(statistic):
    """k of one statistics file in DIR_DATA (bispec: the (3, n_triangles) k123 array)."""
    root = Path(paths.DIR_DATA) / STAT_FILES[statistic]
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        with os.scandir(d) as it:
            fn = next((e.path for e in it if e.name.endswith(".npy")), None)
        if fn is not None:
            break
    else:
        raise FileNotFoundError(f"no {statistic} files under {root}")
    if statistic == "bispec":
        return np.asarray(np.load(fn, allow_pickle=True).item()["k123"]), fn
    k, _, _, _ = data_loader.load_pk(fn)
    return np.asarray(k), fn


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--overwrite", action="store_true", help="replace masks that differ")
    args = p.parse_args()

    out_dir = data_loader.masks_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    for statistic in STAT_FILES:
        k, fn = sample_k(statistic)
        print(f"\n{statistic}: k from {fn}")
        top = data_loader._bin_top_edges(statistic, k)
        for kmax in data_loader.K_CUTS:
            tag = data_loader.kcut_tag(statistic, kmax)
            mask = data_loader.compute_kcut_mask(statistic, tag, k)
            path = data_loader.mask_path(statistic, tag)
            if path.is_file():
                old = np.loadtxt(path, dtype=int).astype(bool)
                if old.shape == mask.shape and np.array_equal(old, mask):
                    status = "exists"
                elif args.overwrite:
                    status = "REPLACED"
                else:
                    raise SystemExit(f"{path} differs from the computed mask; rerun with --overwrite")
            else:
                status = "written"
            if status != "exists":
                np.savetxt(path, mask.astype(int), fmt="%i")
            kept_top = top[..., mask].max()
            print(f"  {tag:10s} {mask.sum():4d}/{mask.size} bins, top edge {kept_top:.4f}  [{status}]")
    print(f"\nmasks in {out_dir}")


if __name__ == "__main__":
    main()
