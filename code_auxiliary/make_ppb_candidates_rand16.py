"""make_ppb_candidates_rand16.py

Candidates for the P+P_gm+B 3rd ensemble setting (2026-11-23). The current setting 3 (bydsd9rq, 128 wide) is
0.43-0.89 nats behind the best member at every kb; the next-best random trials are also 128 wide. Before adopting
one, train it where the old setting 3 failed worst (kb0.27, kb0.37; kpgm0.37, kp0.37), seed 0, and apply the 0.4-nat
rule against the members already there (notebooks/2026-11-19_v2_checks.ipynb B3).

  --nth 3 4  : best_runs.txt lines 3, 4 (choose_best_run.py --n-passing 6; expected 1npeh1qm, 1qop6100), models
               _best-rand16-topkpgm0.37_nbest{n}
Tests: SHAMe at the 3 densities, phase-matched mocks (individual + mean), so a candidate that passes is usable as is.

Run with env_v2.sh sourced (seconds; login node is fine), after best_runs.txt has the lines:
  python make_ppb_candidates_rand16.py --nth 3 4
  sbatch --job-name=train_ppb_candidates_rand16 --array=0-<n-1> jobs/job_train_v2.s jobs/lists/ppb_candidates_rand16.txt
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fresh_fid_rand16 as mff  # noqa: E402  (sets sys.path / cwd for code/)
import generate_config_inference as gci  # noqa: E402

KBS = (0.27, 0.37)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nth", nargs="+", type=int, required=True)
    a = ap.parse_args()
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    kw = gci.resolve_train_tag_bundle("_p5_n10000", "noisy")
    lines = [mff.line(mff.PPB, ["_kp0.37", "_kpgm0.37", f"_kb{kb:.2f}"], nth, 0, kw) for nth in a.nth for kb in KBS]
    fn = mff.DIR_LISTS / "ppb_candidates_rand16.txt"
    fn.write_text("\n".join(lines) + "\n")
    print(f"wrote {fn} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
