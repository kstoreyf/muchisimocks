"""bench_train_threads.py

Time sbi NPE training (MAF, 10 transforms) per (hidden width, batch size, thread count) on CPU.
Synthetic data with the v2 pk+pgm shapes (59 features, 11 params), 32k rows; sbi's
max_num_epochs=1 actually runs 2 epochs, so times are per 2 epochs incl. validation.
Findings 2026-11-16: at batch 32-64, more threads are slower (per-step overhead dominates).
Run via code_auxiliary/jobs/job_bench_threads.s.
"""
import argparse
import time

import numpy as np
import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

ap = argparse.ArgumentParser()
ap.add_argument('--threads', type=int, nargs='+', default=[1, 4])
ap.add_argument('--bs', type=int, nargs='+', default=[64, 256, 1024])
ap.add_argument('--hidden', type=int, nargs='+', default=[256, 128])
args = ap.parse_args()

N, NX, NTH = 32000, 59, 11
rng = np.random.default_rng(0)
th = rng.uniform(-1, 1, (N, NTH))
theta = torch.tensor(th, dtype=torch.float32)
x = torch.tensor(rng.normal(size=(N, NX)) + th @ rng.normal(size=(NTH, NX)), dtype=torch.float32)
prior = BoxUniform(-torch.ones(NTH), torch.ones(NTH))

for hidden in args.hidden:
    for bs in args.bs:
        for nt in args.threads:
            torch.set_num_threads(nt)
            torch.manual_seed(0)
            inf = NPE(prior=prior, density_estimator=posterior_nn(model='maf', hidden_features=hidden, num_transforms=10))
            inf.append_simulations(theta, x)
            t0 = time.time()
            inf.train(max_num_epochs=1, training_batch_size=bs, learning_rate=3e-5,
                      validation_fraction=0.1, show_train_summary=False)
            dt = time.time() - t0
            print(f'\nRESULT hidden={hidden:3d} bs={bs:4d} threads={nt:2d}: {dt:6.1f} s (2 epochs, 32k rows)'
                  f' -> ~{dt * 10 / 2 / 60:5.2f} min per full 320k epoch', flush=True)
