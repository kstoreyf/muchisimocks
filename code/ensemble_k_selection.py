#!/usr/bin/env python3
"""Choose ensemble size K by validation log-prob (equal-weight mixture).

Rank models by min validation loss, evaluate mixture log-prob for K=1..N on a
fixed held-out split. Caches validation arrays under sweep_root/.

Usage (GPU node recommended):
  conda activate benv
  python /scratch/kstoreyf/ensemble_k_selection.py

Recovery copy of wiped notebook:
  /scratch/kstoreyf/2026-06-04_sweep_top5_ensemble_shame.ipynb.recover
"""

from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import logsumexp

REPO = Path("/home/kstoreyf/muchisimocks")
sys.path.insert(0, str(REPO / "code"))

import data_loader  # noqa: E402
import paths  # noqa: E402
import utils_inference  # noqa: E402
from choose_best_run import (  # noqa: E402
    BEST_RUN_LOG_CSV,
    build_sweep_tag_inf,
    sweep_root_for_tag_inf,
)
from generate_config_inference import (  # noqa: E402
    resolve_train_tag_bundle,
    tags_mask_for_sweep,
)

NOISE_MODE = "noisy"
BX = 32
N_TRAIN = 10_000
TAG_SWEEP = "-rand30"
STATISTICS = ["pk", "bispec", "pgm"]
VALIDATION_FRACTION = 0.1
VAL_SPLIT_SEED = 42
LOG_PROB_BATCH_SIZE = 512
VAL_SUBSAMPLE = 5000  # subsample val for CPU; set None for full 32k (use GPU)


def load_ranked_runs(sweep_root: Path) -> list[dict]:
    rows = []
    with open(sweep_root / BEST_RUN_LOG_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("run_name") or "").strip()
            stall = (row.get("pass_stall_test") or "").strip().lower()
            if not name or stall != "true":
                continue
            loss_s = (row.get("best_val_loss") or "").strip()
            rows.append(
                {
                    "run_name": name,
                    "best_val_loss": float(loss_s) if loss_s else None,
                }
            )
    rows.sort(key=lambda r: (r["best_val_loss"] is None, r["best_val_loss"] or float("inf")))
    return [r for r in rows if (sweep_root / r["run_name"] / "posterior.p").is_file()]


def load_val_arrays(
    *,
    sweep_root: Path,
    train_bundle: dict,
    tags_mask: list[str],
) -> tuple[np.ndarray, list[np.ndarray]]:
    cache_fn = sweep_root / "ensemble_val_cache.npz"
    if cache_fn.is_file():
        print(f"Loading cache {cache_fn}", flush=True)
        cached = np.load(cache_fn, allow_pickle=True)
        theta_val = cached["theta_val"]
        y_val = [cached[f"y_val_{i}"] for i in range(len(STATISTICS))]
        return theta_val, y_val

    print("Building val cache (loads n_train cosmologies only)...", flush=True)
    _, _, _, _, random_ints_cosmo, _ = data_loader.load_params(
        train_bundle["tag_params"], train_bundle["tag_biasparams"], bx=BX
    )
    cosmo_indices = random_ints_cosmo[:N_TRAIN]
    _, y, _, idxs_params, params_df, _, biasparams_df, _, random_ints_cosmo, _ = (
        data_loader.load_data(
            "muchisimocks",
            STATISTICS,
            train_bundle["tag_params"],
            train_bundle["tag_biasparams"],
            tag_noise=train_bundle["tag_noise"],
            tags_mask=tags_mask,
            bx=BX,
            cosmo_indices=cosmo_indices,
        )
    )
    theta, pnames = data_loader.param_dfs_to_theta(idxs_params, params_df, biasparams_df)
    theta, pnames = utils_inference.reparameterize_theta(theta, pnames)
    exclude = {"hubble", "omega_baryon", "ns"}
    keep = [i for i, pn in enumerate(pnames) if pn not in exclude]
    theta = theta[:, keep]

    idxs_cosmo = random_ints_cosmo[:N_TRAIN]
    idxs_all = np.arange(len(y[0]))
    idxs_train = idxs_all[np.isin(idxs_params[:, 0], idxs_cosmo)]
    theta_train = theta[idxs_train]
    y_train = [yi[idxs_train] for yi in y]
    print(f"theta_train {theta_train.shape}", flush=True)

    n_val = int(round(VALIDATION_FRACTION * len(theta_train)))
    val_idx = np.random.default_rng(VAL_SPLIT_SEED).permutation(len(theta_train))[:n_val]
    theta_val = theta_train[val_idx]
    y_val = [yi[val_idx] for yi in y_train]
    print(f"val set: {len(val_idx)} sims", flush=True)

    save = {"theta_val": theta_val, "val_idx": val_idx}
    for i, block in enumerate(y_val):
        save[f"y_val_{i}"] = block
    np.savez(cache_fn, **save)
    print(f"Saved {cache_fn}", flush=True)
    return theta_val, y_val


def scale_y(y_blocks: list[np.ndarray], run_dir: Path) -> np.ndarray:
    out = np.empty((len(y_blocks[0]), 0), dtype=np.float32)
    for stat, yi in zip(STATISTICS, y_blocks):
        with open(run_dir / f"scaler_y_{stat}.p", "rb") as f:
            scaler = pickle.load(f)
        out = np.concatenate([out, scaler.scale(yi).astype(np.float32)], axis=1)
    return out


def batched_log_prob(posterior, theta_np, x_np, device, batch_size=LOG_PROB_BATCH_SIZE):
    chunks = []
    n = len(theta_np)
    for start in range(0, n, batch_size):
        th = torch.as_tensor(theta_np[start : start + batch_size], dtype=torch.float32, device=device)
        xb = torch.as_tensor(x_np[start : start + batch_size], dtype=torch.float32, device=device)
        with torch.no_grad():
            # Paired (theta_i, x_i): theta shape (1, batch, n_params), x shape (batch, n_x)
            lp = posterior.log_prob_batched(
                th.unsqueeze(0), x=xb, norm_posterior=False
            )
        chunks.append(lp.squeeze(0).detach().cpu().numpy())
    return np.concatenate(chunks)


def main() -> None:
    tags_mask = tags_mask_for_sweep(STATISTICS)
    train_bundle = resolve_train_tag_bundle("_p5_n10000", NOISE_MODE)
    tag_inf = build_sweep_tag_inf(
        data_mode="muchisimocks",
        statistics=STATISTICS,
        tags_mask=tags_mask,
        tag_params=train_bundle["tag_params"],
        tag_biasparams=train_bundle["tag_biasparams"],
        tag_noise=train_bundle["tag_noise"],
        reparameterize=True,
        tag_sweep=TAG_SWEEP,
    )
    sweep_root = sweep_root_for_tag_inf(tag_inf)
    rows = load_ranked_runs(sweep_root)
    print(f"{len(rows)} stall-passing models with posterior.p", flush=True)

    theta_val, y_val = load_val_arrays(
        sweep_root=sweep_root, train_bundle=train_bundle, tags_mask=tags_mask
    )
    if VAL_SUBSAMPLE is not None and VAL_SUBSAMPLE < len(theta_val):
        sub = np.random.default_rng(VAL_SPLIT_SEED + 1).choice(
            len(theta_val), size=VAL_SUBSAMPLE, replace=False
        )
        theta_val = theta_val[sub]
        y_val = [yi[sub] for yi in y_val]
        print(f"Subsampled val -> {len(theta_val)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    logp_stack = []
    for i, row in enumerate(rows):
        run_dir = sweep_root / row["run_name"]
        with open(run_dir / "posterior.p", "rb") as f:
            post = pickle.load(f)
        x_val = scale_y(y_val, run_dir)
        lp = batched_log_prob(post, theta_val, x_val, device)
        logp_stack.append(lp)
        print(
            f"  [{i+1}/{len(rows)}] {row['run_name']}  mean log p={lp.mean():.4f}  "
            f"stored ell_val={row['best_val_loss']:.4g}",
            flush=True,
        )

    logp_stack = np.stack(logp_stack, axis=0)
    ks = np.arange(1, logp_stack.shape[0] + 1)
    mean_logp_k = np.array(
        [logsumexp(logp_stack[:k], axis=0).mean() - np.log(k) for k in ks]
    )
    best_k = int(ks[np.argmax(mean_logp_k)])
    print(f"\nBest K={best_k}  mean log p={mean_logp_k[best_k - 1]:.4f}", flush=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, mean_logp_k, "o-", lw=1.5, ms=5)
    ax.axvline(best_k, color="C3", ls="--", lw=1, label=f"best K={best_k}")
    ax.set_xlabel("ensemble size K")
    ax.set_ylabel(r"mean validation $\log p(\theta\mid x)$")
    ax.set_title("Equal-weight mixture log-prob vs K")
    ax.set_xticks(ks)
    ax.legend()
    fig.tight_layout()
    out_png = sweep_root / "ensemble_k_selection.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
