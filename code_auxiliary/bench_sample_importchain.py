"""bench_sample_importchain.py

Production tests sample ~60x slower than bench_sample_threads.py for the same model (1 obs, 12000 draws:
660 s vs ~11 s). Replicate run_inference.py's import chain (OMP_NUM_THREADS=1, TF, data loaders) and time
raw draws for 1 obs, first as-is, then after explicitly torch.set_num_threads(1).
"""
import os
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
os.chdir(Path(__file__).resolve().parent.parent / "code")
import run_inference  # noqa: E402,F401  (same imports as production)
import torch  # noqa: E402

print("torch threads intra/inter:", torch.get_num_threads(), torch.get_num_interop_threads(),
      "| cpus allowed:", len(os.sched_getaffinity(0)), "| torch config:", torch.__config__.parallel_info().split("\n")[:6])
posterior = pickle.load(open(Path(sys.argv[1]) / "posterior.p", "rb"))
est = posterior.posterior_estimator
x = torch.zeros(1, int(est.condition_shape[-1]))

def timed(label):
    with torch.no_grad():
        t0 = time.time()
        est.sample((12000,), condition=x)
    print(f"{label}: 1 obs x 12000 raw draws {time.time() - t0:.2f}s", flush=True)

timed("as production (import chain)")
torch.set_num_threads(1)
timed("after torch.set_num_threads(1)")
