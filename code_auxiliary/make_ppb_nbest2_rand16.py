"""make_ppb_nbest2_rand16.py

Retrain P+P_gm+B setting 3 (``_best-rand16-topkpgm0.37_nbest2``) everywhere after its sweep trial 0jwzywhf was
retired (2026-11-23, retire_sweep_trial.py: fails the 0.4-nat rule at kb 0.27-0.37 even with a new seed). The old
4th-best trial moves into best_runs.txt line 2, so the top-3 naming and ensemble code are unchanged; the old
nbest2 dirs were moved to results_sbi/retired_sweep_trials/. Submit after choose_best_run.py has rewritten
best_runs.txt (training reads line 2 at run time).

One list, kpgmslice + kbslice members of setting 3, same layout as make_kbslice_top3_rand16.py and
make_kpgmslice_top3_rand16.py:
  - kb slice (kpgm0.37): all 9 kb; seed 1 at kb0.24 (fresh fiducial member, + coverage inner500), seed 0 elsewhere;
    plus seed 0 at kb0.24 (= the copied sweep trial, + coverage)
  - kpgm slice (kb0.24): the 8 kpgm != 0.37, seed 0
Tests per model: SHAMe at the 3 densities, phase-matched mocks at the 3 densities (individual + mean).

Run with env_v2.sh sourced (seconds; login node is fine):
  python make_ppb_nbest2_rand16.py
  sbatch --job-name=train_ppb_nbest2_rand16 --array=0-17 [--dependency=afterok:<choosebest job>] \
      jobs/job_train_v2.s jobs/lists/ppb_nbest2_rand16.txt
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fresh_fid_rand16 as mff  # noqa: E402  (sets sys.path / cwd for code/)
import data_loader  # noqa: E402
import generate_config_inference as gci  # noqa: E402

KP, KPGM_SWEEP, KB_SWEEP, NTH = 0.37, 0.37, 0.24, 2


def masks(kpgm, kb):
    return [data_loader.kcut_tag("pk", KP), data_loader.kcut_tag("pgm", kpgm), data_loader.kcut_tag("bispec", kb)]


def main():
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    assert gci.SWEEP_MASKS["pgm"] == data_loader.kcut_tag("pgm", KPGM_SWEEP), "expects the kpgm0.37 fiducial"
    kw = gci.resolve_train_tag_bundle("_p5_n10000", "noisy")
    lines = [mff.line(mff.PPB, masks(KPGM_SWEEP, kb), NTH, int(kb == KB_SWEEP), kw, coverage=kb == KB_SWEEP)
             for kb in data_loader.K_CUTS]
    lines += [mff.line(mff.PPB, masks(KPGM_SWEEP, KB_SWEEP), NTH, 0, kw, coverage=True)]
    lines += [mff.line(mff.PPB, masks(k, KB_SWEEP), NTH, 0, kw) for k in data_loader.K_CUTS if k != KPGM_SWEEP]
    fn = mff.DIR_LISTS / "ppb_nbest2_rand16.txt"
    fn.write_text("\n".join(lines) + "\n")
    print(f"wrote {fn} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
