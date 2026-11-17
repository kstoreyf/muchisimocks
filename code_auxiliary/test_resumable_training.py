"""End-to-end test of checkpointed sweep training (sbi_model TRAIN_CHECKPOINT) on synthetic data.

Uses a throwaway W&B project (MUCHISIMOCKS_WANDB_PROJECT, default muchisimocks-sbi-test) and a
throwaway results tree. Run the phases as separate processes (job_test_resumable_training.s):
  python test_resumable_training.py stop     # deadline 5 min out -> stops after 1 chunk, exit 3
  python test_resumable_training.py resume   # resumes the same W&B run id, trains to the end
  python test_resumable_training.py full     # sweep full, nothing to resume -> no-op
  python test_resumable_training.py check    # assertions on W&B + local files
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
os.environ.setdefault("MUCHISIMOCKS_WANDB_PROJECT", "muchisimocks-sbi-test")
os.environ.setdefault("MUCHISIMOCKS_RESULTS", "/scratch/kstoreyf/muchisimocks/test_resumable/results")

import sbi_model  # noqa: E402

ROOT = Path(os.environ["MUCHISIMOCKS_RESULTS"]).parent
TAG = "_test_resumable"
YAML = ROOT / "config_test_resumable.yaml"


def make_model():
    rng = np.random.default_rng(0)
    n_cosmo, bx = 400, 4
    theta = rng.uniform(-1, 1, size=(n_cosmo * bx, 2))
    k = np.linspace(0.1, 1, 8)
    y = np.exp(theta[:, :1] * k + 0.5 * theta[:, 1:] * k**2) * (1 + 0.05 * rng.normal(size=(len(theta), len(k))))
    groups = np.repeat(np.arange(n_cosmo), bx)
    bounds = {"a": [-1, 1], "b": [-1, 1]}
    return sbi_model.SBIModel(
        theta_train=theta, y_train_unscaled=[y], statistics=["pk"], param_names=["a", "b"],
        dict_bounds=bounds, dict_bounds_physical=bounds, run_mode="sweep", tag_sbi=TAG,
        sweep_name=TAG, sweep_num_runs=1, wandb_config_yaml_path=str(YAML), groups_train=groups,
    )


def main(phase):
    ROOT.mkdir(parents=True, exist_ok=True)
    if phase == "stop":
        assert not YAML.exists(), f"stale test state; remove {ROOT}"
        YAML.write_text("wandb_sweep_id: null\n")
        os.environ["MUCHISIMOCKS_TRAIN_DEADLINE"] = str(time.time() + 300)
    if phase in ("stop", "resume", "full"):
        m = make_model()
        m.run(max_epochs=60)
        print(f"PHASE {phase}: training_incomplete={m.training_incomplete}", flush=True)
        sys.exit(sbi_model.EXIT_TRAINING_INCOMPLETE if m.training_incomplete else 0)
    if phase == "check":
        import wandb
        import yaml
        sweep_path = yaml.safe_load(YAML.read_text())["wandb_sweep_id"]
        runs = list(wandb.Api().sweep(sweep_path).runs)
        print("sweep runs:", [(r.id, r.state) for r in runs])
        assert len(runs) == 1, "expected exactly one run (resumed, not redrawn)"
        r = runs[0]
        assert r.state == "finished", r.state
        run_dir = Path(os.environ["MUCHISIMOCKS_RESULTS"]) / "results_sbi" / f"sbi{TAG}" / r.id
        assert (run_dir / "posterior.p").is_file()
        assert not (run_dir / sbi_model.TRAIN_CHECKPOINT).exists(), "checkpoint should be removed"
        assert not (run_dir / sbi_model.TRAIN_INCOMPLETE_MARKER).exists()
        with open(run_dir / "inference.p", "rb") as f:
            inf = pickle.load(f)
        n_ep = len(inf._summary["validation_loss"])
        assert n_ep > sbi_model.TRAIN_CHUNK_EPOCHS + 1, f"only {n_ep} epochs: did it resume?"
        print("best_validation_loss summary:", r.summary.get("best_validation_loss"),
              "| epochs:", n_ep, "| min val:", min(inf._summary["validation_loss"]))
        print("CHECK OK")


if __name__ == "__main__":
    main(sys.argv[1])
