"""make_kpgmslice_top3_rand16.py

k_max^{P_gm} slice of P+P_gm+B with the NEW fiducial hyperparameters (top 3 of the kpgm0.37/kb0.24 sweep, models
``_best-rand16-topkpgm0.37[_nbest{n}]``), kp0.37 and kb0.24 throughout, kpgm over data_loader.K_CUTS. Redoes the
old-hyperparameter slice (make_kpgm37_checks_rand16.py, notebooks/2026-11-19_v2_checks.ipynb §6) so the paper's kpgm
figure uses the fiducial hyperparameters; the old slice stays as the hyperparameter-transfer appendix check.

Every point is a fresh training: at the sweep cut (kpgm0.37) the members are seed 1 of each setting, which already
exist from the kb slice (make_kbslice_top3_rand16.py), so only kpgm != 0.37 is listed (8 cuts x 3 settings, seed 0).

Tests per model: SHAMe at the 3 densities, phase-matched mocks at the 3 densities (individual + mean).

Run with env_v2.sh sourced (seconds; login node is fine):
  python make_kpgmslice_top3_rand16.py
  sbatch --job-name=train_kpgmslice_top3_newhp --array=0-<n-1> jobs/job_train_v2.s jobs/lists/kpgmslice_top3_newhp_rand16.txt
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fresh_fid_rand16 as mff  # noqa: E402  (sets sys.path / cwd for code/)
import data_loader  # noqa: E402
import generate_config_inference as gci  # noqa: E402

KP, KB, KPGM_SWEEP = 0.37, 0.24, 0.37


def main():
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    assert gci.SWEEP_MASKS["pgm"] == data_loader.kcut_tag("pgm", KPGM_SWEEP), "expects the kpgm0.37 fiducial"
    kw = gci.resolve_train_tag_bundle("_p5_n10000", "noisy")
    lines = [mff.line(mff.PPB, [data_loader.kcut_tag("pk", KP), data_loader.kcut_tag("pgm", k),
                                data_loader.kcut_tag("bispec", KB)], nth, 0, kw)
             for k in data_loader.K_CUTS if k != KPGM_SWEEP for nth in (0, 1, 2)]
    fn = mff.DIR_LISTS / "kpgmslice_top3_newhp_rand16.txt"
    fn.write_text("\n".join(lines) + "\n")
    print(f"wrote {fn} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
