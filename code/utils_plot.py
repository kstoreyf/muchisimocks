"""
Plot/notebook helpers for muchisimocks inference.

This module centralizes "tag construction" logic so notebooks can stay small.
It is designed to work with the current codebase where noise parameters live
inside the bias parameter dataframe (no separate Anoise tag needed).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from pathlib import Path

import data_loader
import utils


def _ensure_list(x: Union[str, Sequence[str], None], n: int) -> List[Optional[str]]:
    """Normalize `x` to a list of length `n`."""
    if x is None:
        return [None] * n
    if isinstance(x, str):
        return [x] * n
    x_list = list(x)
    if len(x_list) != n:
        raise ValueError(f"Expected length {n}, got {len(x_list)}.")
    return x_list


def get_param_names_key():
    """
    Shared parameter-name lists used for 1D coverage plots.

    Returns
    -------
    (param_names_key, param_names_key_rp)
        - `param_names_key`: unreparameterized names
        - `param_names_key_rp`: reparameterized names (sigma8*bias combinations)
    """
    param_names_key = ["omega_cold", "sigma8_cold", "b1"]
    param_names_key_rp = ["omega_cold", "sigma8_cold", "sigma8_cold_x_b1"]
    return param_names_key, param_names_key_rp


def setup_inference_tags(
    data_mode: str,
    tag_params: str,
    tag_biasparams: str,
    statistics_arr: Sequence[Sequence[str]],
    bx: int,
    tag_noise: Optional[str] = None,
    tag_reparam: str = "_rp",
    n_train: int = 10000,
    tags_mask: Optional[Sequence[str]] = None,
):
    """
    Construct SBI inference tags for the given statistics and training setup.

    Returns
    -------
    tags_inf, labels, colors, tag_stats_arr
    """
    tag_stats_arr = [f"_{'_'.join(statistics)}" for statistics in statistics_arr]

    if tags_mask is None:
        tags_mask = [""] * len(tag_stats_arr)
    if len(tags_mask) != len(tag_stats_arr):
        raise ValueError("tags_mask must be the same length as statistics_arr.")

    tag_num = f"_bx{bx}_ntrain{n_train}"

    tags_inf = []
    for i, tag_stats in enumerate(tag_stats_arr):
        # tag_stats already includes the leading underscore.
        if tag_noise is None:
            tags_inf.append(f"_{data_mode}{tag_stats}{tags_mask[i]}{tag_params}{tag_biasparams}{tag_reparam}{tag_num}")
        else:
            tags_inf.append(
                f"_{data_mode}{tag_stats}{tags_mask[i]}{tag_params}{tag_biasparams}{tag_noise}{tag_reparam}{tag_num}"
            )

    labels = [utils.get_stat_label(stat) for stat in statistics_arr]
    colors = utils.get_stat_colors(statistics_arr)

    return tags_inf, labels, colors, tag_stats_arr


def load_training_params(
    tag_params: str,
    tag_biasparams: str,
    bx: Optional[int] = None,
):
    """
    Load training set parameters and determine which parameters vary.

    Noise parameters are included inside the bias parameter dataframe in the
    current codebase, so we return only:
      - cosmo_param_names_vary
      - bias_param_names_vary
      - param_names_vary = cosmo + bias
    """
    (
        params_df,
        _param_dict_fixed,
        biasparams_df,
        _biasparams_dict_fixed,
        _random_ints,
        _random_ints_bias,
    ) = data_loader.load_params(tag_params, tag_biasparams, bx=bx)

    cosmo_param_names_vary: List[str] = []
    bias_param_names_vary: List[str] = []

    if params_df is not None:
        cosmo_param_names_vary = list(params_df.columns.tolist())

    if biasparams_df is not None:
        # Filter out nested-bias metadata columns (but keep actual physics params,
        # including additive/multiplicative noise parameters, which now live here).
        nested_meta_cols = getattr(data_loader, "NESTED_META_COLS", ["idx_cosmo", "nest_layer"])
        bias_param_names_vary = [c for c in biasparams_df.columns.tolist() if c not in nested_meta_cols]
        if len(bias_param_names_vary) == 0:
            bias_param_names_vary = [
                c
                for c in biasparams_df.columns.tolist()
                if c not in ("idx_cosmo", "nest_layer")
            ]

    param_names_vary = cosmo_param_names_vary + bias_param_names_vary
    return cosmo_param_names_vary, bias_param_names_vary, param_names_vary


def setup_test_tags(
    data_mode: str,
    tag_params_test: str,
    tags_biasparams_test: Union[str, Sequence[str]],
    tag_stats_arr: Sequence[str],
    n_test_eval: Optional[int] = None,
    tag_datagen_test: str = "",
    tags_mask_test: Optional[Sequence[str]] = None,
    tag_noise_test: Optional[str] = None,
):
    """
    Construct tag_test strings for SBI sample loading.

    Notes
    -----
    - `tag_noise_test` corresponds to the additive-noise portion of the tag.
    - Multiplicative noise parameters are expected to already be included in
      `tags_biasparams_test`.
    """
    n = len(tag_stats_arr)
    tags_biasparams_test_list = _ensure_list(tags_biasparams_test, n)

    if tags_mask_test is None:
        tags_mask_test = [""] * n
    if len(tags_mask_test) != n:
        raise ValueError("tags_mask_test must have the same length as tag_stats_arr.")

    noise_tag = "" if tag_noise_test is None else tag_noise_test

    tags_data_test = []
    for i in range(n):
        base = f"_{data_mode}{tag_stats_arr[i]}{tags_mask_test[i]}{tag_params_test}{tags_biasparams_test_list[i]}{noise_tag}{tag_datagen_test}"
        if n_test_eval is not None:
            base = f"{base}_neval{n_test_eval}"
        tags_data_test.append(base)
    return tags_data_test


def setup_shame_mock_test_tags(
    tag_stats_arr: Sequence[str],
    tag_mock: str = "_nbar0.00022",
    data_mode_test: str = "shame",
    data_mode: Optional[str] = None,
):
    """
    Construct OOD SHAMe test tags used for SBI sample loading.

    Accepts `data_mode` as an alias for `data_mode_test` for notebook
    compatibility.
    """
    if data_mode is not None:
        data_mode_test = data_mode
    return [f"_{data_mode_test}{tag_stats}{tag_mock}" for tag_stats in tag_stats_arr]


def load_test_predictions(
    tags_inf: Sequence[str],
    tags_data_test: Sequence[str],
    tag_params_test: str,
    tags_biasparams_test: Union[str, Sequence[str]],
    cosmo_param_names_vary: Sequence[str],
    bias_param_names_vary: Sequence[str],
    param_names_vary: Sequence[str],
    tag_reparam: str = "_rp",
    n_test_eval: Optional[int] = None,
    param_names_show: Optional[Sequence[str]] = None,
):
    """
    Load SBI predicted moments and the corresponding true parameter values.

    This mirrors the older notebook logic, but uses the updated API
    (no tag_Anoise, and noise parameters are part of bias parameters).
    """
    n = len(tags_inf)
    tags_biasparams_test_list = _ensure_list(tags_biasparams_test, n)

    if param_names_show is None:
        _theta0, _cov0, param_names = utils.get_moments_test_sbi(tags_inf[0], tag_test=tags_data_test[0])
        param_names_show = list(param_names)

    theta_true_arr, theta_pred_arr, vars_pred_arr, covs_pred_arr = [], [], [], []

    for i in range(n):
        _theta_test_pred, _covs_test_pred, param_names_chain = utils.get_moments_test_sbi(
            tags_inf[i], tag_test=tags_data_test[i], param_names=param_names_show
        )

        theta_test = data_loader.load_theta_test(
            tag_params_test,
            tags_biasparams_test_list[i],
            cosmo_param_names_vary=list(cosmo_param_names_vary),
            bias_param_names_vary=list(bias_param_names_vary),
        )

        # Ensure shape is [n_samples, n_params] for downstream indexing.
        if theta_test.ndim == 1:
            n_samples = n_test_eval if n_test_eval is not None else _theta_test_pred.shape[0]
            theta_test = np.tile(theta_test, (n_samples, 1))
        elif n_test_eval is not None:
            theta_test = theta_test[:n_test_eval]

        if tag_reparam:
            theta_test, param_names_test_reparam = utils.reparameterize_theta(theta_test, list(param_names_vary))
        else:
            param_names_test_reparam = list(param_names_vary)

        # Extract only the parameters we want to show.
        theta_true_inf, theta_pred_inf, vars_pred_inf = [], [], []
        for pn in param_names_show:
            if pn in param_names_chain:
                idx = list(param_names_chain).index(pn)
                theta_pred_inf.append(_theta_test_pred[:, idx])

                if _covs_test_pred.ndim == 2:
                    vars_pred_inf.append(np.full(theta_test.shape[0], _covs_test_pred[idx, idx]))
                else:
                    vars_pred_inf.append(_covs_test_pred[:, idx, idx])

                if pn in param_names_test_reparam:
                    idx_test = list(param_names_test_reparam).index(pn)
                    theta_true_inf.append(theta_test[:, idx_test])
                else:
                    theta_true_inf.append(np.full(theta_test.shape[0], np.nan))
            else:
                theta_true_inf.append(np.full(theta_test.shape[0], np.nan))
                theta_pred_inf.append(np.full(theta_test.shape[0], np.nan))
                vars_pred_inf.append(np.full(theta_test.shape[0], np.nan))

        theta_true_arr.append(np.array(theta_true_inf).T)
        theta_pred_arr.append(np.array(theta_pred_inf).T)
        vars_pred_arr.append(np.array(vars_pred_inf).T)
        covs_pred_arr.append(_covs_test_pred)

    return (
        np.array(theta_true_arr),
        np.array(theta_pred_arr),
        np.array(vars_pred_arr),
        np.array(covs_pred_arr),
        list(param_names_show),
    )


def _sbi_samples_paths(tag_inf: str, tag_test: str) -> Tuple[Path, Path]:
    """
    Build the paths that `utils.get_samples_sbi` would load.

    Returned paths are (pred.npy, pred_inprogress.npy).
    """
    # Build absolute paths from this file's location so callers don't have to
    # care what the current working directory is.
    repo_root = Path(__file__).resolve().parents[1]
    dir_sbi = repo_root / "results" / "results_sbi" / f"sbi{tag_inf}"
    fn_pred = dir_sbi / f"samples_test{tag_test}_pred.npy"
    fn_inprogress = dir_sbi / f"samples_test{tag_test}_pred_inprogress.npy"
    return fn_pred, fn_inprogress


def sbi_samples_exist(tag_inf: str, tag_test: str) -> bool:
    """Whether SBI posterior samples for this tag_test exist (pred or in-progress)."""
    fn_pred, fn_inprogress = _sbi_samples_paths(tag_inf, tag_test)
    return fn_pred.exists() or fn_inprogress.exists()


def filter_available_chains(
    tags_inf: Sequence[str],
    tags_test: Sequence[str],
    inf_methods: Sequence[str],
    labels: Sequence[str],
    colors: Sequence[str],
):
    """Filter to only those (tag_inf, tag_test) pairs that exist on disk."""
    keep = [sbi_samples_exist(ti, tt) for ti, tt in zip(tags_inf, tags_test)]
    if not any(keep):
        return [], [], [], [], []
    tags_inf_kept = [t for t, k in zip(tags_inf, keep) if k]
    tags_test_kept = [t for t, k in zip(tags_test, keep) if k]
    inf_methods_kept = [m for m, k in zip(inf_methods, keep) if k]
    labels_kept = [l for l, k in zip(labels, keep) if k]
    colors_kept = [c for c, k in zip(colors, keep) if k]
    return tags_inf_kept, tags_test_kept, inf_methods_kept, labels_kept, colors_kept

