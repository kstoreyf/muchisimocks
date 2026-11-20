"""bench_sample_threads.py

Time posterior sampling (sbi_model.sample_batched_isolated, 10000 draws per obs) for a trained model
per CPU thread count, and on GPU if one is visible. Training is fastest single-threaded
(bench_train_threads.py), but the rand16 networks are wide (MAF hidden 512), so sampling may differ.
Run via code_auxiliary/jobs/job_bench_sample.s.
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
from sbi_model import sample_batched_isolated  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dir_sbi")
ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16])
ap.add_argument("--n-obs", type=int, default=5)
ap.add_argument("--n-samples", type=int, default=10000)
args = ap.parse_args()

posterior = pickle.load(open(Path(args.dir_sbi) / "posterior.p", "rb"))
est = posterior.posterior_estimator
x_dim = int(est.condition_shape[-1])
x = torch.zeros(args.n_obs, x_dim)   # scaled data space: zeros = training mean
print(f"{args.dir_sbi}\nx_dim={x_dim}, {args.n_obs} obs x {args.n_samples} samples")

for nt in args.threads:
    torch.set_num_threads(nt)
    t0 = time.time()
    sample_batched_isolated(posterior, x, args.n_samples, verbose=True)
    dt = time.time() - t0
    print(f"cpu threads={nt:2d}: {dt:7.1f}s total, {dt / args.n_obs:6.1f}s per obs", flush=True)

# support check alone, on 5 x 12000 draws
from sbi.utils.sbiutils import within_support  # noqa: E402
with torch.no_grad():
    cand = est.sample((int(1.2 * args.n_samples),), condition=x)
t0 = time.time()
ok = within_support(posterior.prior, cand)
print(f"within_support on {tuple(cand.shape)}: {time.time() - t0:.2f}s, accept frac {ok.float().mean():.3f}", flush=True)

if torch.cuda.is_available():
    # raw flow draws only (sample_batched_isolated is CPU-only: its support check goes through numpy)
    n_draw = int(1.2 * args.n_samples)
    for dev in ("cpu", "cuda"):
        est.to(dev)
        for n_obs in (args.n_obs, 50):
            xg = torch.zeros(n_obs, x_dim, device=dev)
            with torch.no_grad():
                est.sample((100,), condition=xg[:1])   # warm-up
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                est.sample((n_draw,), condition=xg)
                if dev == "cuda":
                    torch.cuda.synchronize()
            dt = time.time() - t0
            print(f"raw draws {dev} ({torch.get_num_threads()} threads, {n_obs} obs x {n_draw}): "
                  f"{dt:7.2f}s, {dt / n_obs:6.3f}s per obs", flush=True)
