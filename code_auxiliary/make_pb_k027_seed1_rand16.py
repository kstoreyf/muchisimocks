"""make_pb_k027_seed1_rand16.py

Seed-1 retrain of the top P+B network at the overall-k_max point k=0.27 (kp0.27 / kb0.24), user 2026-11-24.
The seed-0 network there has FoM(Omega_m) ~36 at all three n-bar on both SHAMe and the phase-matched mocks (neighbours
43-57, rising with n-bar) and a best val loss ~0.3 nats worse than at k=0.24, so it looks like a bad draw (Fig 9 dip).
Tests: SHAMe at the 3 densities + phase-matched (no coverage).

Run with env_v2.sh sourced (seconds; login node is fine):
  python make_pb_k027_seed1_rand16.py
  sbatch --job-name=train_pb_k027_seed1_rand16 --array=0 jobs/job_train_v2.s jobs/lists/pb_k027_seed1_rand16.txt
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fresh_fid_rand16 as mff  # noqa: E402  (sets sys.path / cwd for code/)
import generate_config_inference as gci  # noqa: E402


def main():
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    kw = gci.resolve_train_tag_bundle("_p5_n10000", "noisy")
    lines = [mff.line(mff.PB, ["_kp0.27", "_kb0.24"], 0, 1, kw)]
    fn = mff.DIR_LISTS / "pb_k027_seed1_rand16.txt"
    fn.write_text("\n".join(lines) + "\n")
    print(f"wrote {fn} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
