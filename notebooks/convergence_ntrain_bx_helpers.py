"""Helper functions for 2026-06-05_convergence_ntrain_bx.ipynb (loaded from cell 3)."""


def build_tag_inf(
    bx: int,
    n_train: int,
    *,
    statistics=None,
    tag_masks=None,
) -> str:
    statistics = statistics if statistics is not None else globals()["statistics"]
    tag_masks = tag_masks if tag_masks is not None else globals()["tag_masks"]
    tag_stats = f"_{'_'.join(statistics)}"
    base = (
        f"_{data_mode}{tag_stats}{tag_masks}{tag_params}{tag_biasparams}"
        f"{tag_noise}{tag_reparam}_bx{bx}_ntrain{n_train}"
    )
    return base + TAG_INF_BEST_SUFFIX


def build_tag_test(*, statistics=None, tag_masks=None) -> str:
    statistics = statistics if statistics is not None else globals()["statistics"]
    tag_masks = tag_masks if tag_masks is not None else globals()["tag_masks"]
    tag_stats = f"_{'_'.join(statistics)}"
    return (
        f"_{data_mode}{tag_stats}{tag_masks}{TAG_PARAMS_TEST}"
        f"{TAG_BIASPARAMS_TEST}{TAG_NOISE_TEST}{TAG_DATAGEN_TEST}"
    )


def build_tag_test_coverage(*, statistics=None, tag_masks=None) -> str:
    statistics = statistics if statistics is not None else globals()["statistics"]
    tag_masks = tag_masks if tag_masks is not None else globals()["tag_masks"]
    tag_stats = f"_{'_'.join(statistics)}"
    return (
        f"_{data_mode}{tag_stats}{tag_masks}{TAG_PARAMS_TEST_COV}"
        f"{TAG_BIASPARAMS_TEST_COV}{TAG_NOISE_TEST_COV}"
    )


def fob_chi2_sqrt_limit(ndims: int) -> float:
    return float(np.sqrt(chi2.ppf(_CHI2_PPF_1SIG, int(ndims))))


def compute_fob(theta_pred_row, cov, param_names, theta_true, param_vary):
    names = list(param_names)
    theta_true_rep, names_rep = ui.reparameterize_theta(theta_true, param_vary)
    names_rep = list(names_rep)
    idx_fob = [names.index(pn) for pn in PARAM_NAMES_FOB]
    mu = np.array(
        [float(theta_pred_row[names.index(pn)]) for pn in PARAM_NAMES_FOB], dtype=float,
    )
    true_fob = np.array(
        [theta_true_rep[names_rep.index(pn)] for pn in PARAM_NAMES_FOB], dtype=float,
    )
    cov_sub = np.asarray(cov, dtype=float)[np.ix_(idx_fob, idx_fob)] + FOB_RIDGE * np.eye(len(idx_fob))
    fob = {}
    for key, i in zip(FOB_KEYS, range(len(FOB_KEYS))):
        sig = np.sqrt(cov_sub[i, i])
        fob[key] = abs(mu[i] - true_fob[i]) / sig if sig > 0 else np.nan
    diff = mu - true_fob
    sign, _ = np.linalg.slogdet(cov_sub)
    cov_inv = np.linalg.inv(cov_sub) if sign > 0 else np.linalg.pinv(cov_sub)
    fob3 = float(np.sqrt(diff @ cov_inv @ diff))
    return fob, fob3


def compute_fom_key_block(cov, param_names):
    names = list(param_names)
    idx_fob = [names.index(pn) for pn in PARAM_NAMES_FOB]
    cov_sub = np.asarray(cov, dtype=float)[np.ix_(idx_fob, idx_fob)]
    fom_marg = {}
    for key, i in zip(FOB_KEYS, range(len(FOB_KEYS))):
        sig = np.sqrt(cov_sub[i, i])
        fom_marg[key] = 1.0 / sig if sig > 0 else np.nan
    det = np.linalg.det(cov_sub)
    fom_3d = 1.0 / np.sqrt(det) if det > 0 else np.nan
    return fom_marg, float(fom_3d)


def load_metrics(
    bx: int,
    n_train: int,
    tag_test=None,
    *,
    statistics=None,
    tag_masks=None,
):
    statistics = statistics if statistics is not None else globals()["statistics"]
    tag_masks = tag_masks if tag_masks is not None else globals()["tag_masks"]
    if tag_test is None:
        tag_test = build_tag_test(statistics=statistics, tag_masks=tag_masks)
    tag_inf = build_tag_inf(bx, n_train, statistics=statistics, tag_masks=tag_masks)
    fn = paths.DIR_RESULTS / "results_sbi" / f"sbi{tag_inf}" / f"samples_test{tag_test}_pred.npy"
    if not fn.is_file():
        return None
    theta_pred, covs_pred, param_names = ui.get_moments_test_sbi(tag_inf, tag_test=tag_test)
    if theta_pred.size == 0:
        return None
    cosmo_vary, bias_vary, param_vary = utils_plot.load_training_params(
        tag_params, tag_biasparams, bx=bx,
    )
    theta_true = data_loader.load_theta_test(
        TAG_PARAMS_TEST, TAG_BIASPARAMS_TEST,
        cosmo_param_names_vary=cosmo_vary, bias_param_names_vary=bias_vary,
    )
    if theta_pred.ndim == 2:
        theta_pred_row = theta_pred[0]
        cov = covs_pred[0] if covs_pred.ndim == 3 else covs_pred
    else:
        theta_pred_row = theta_pred
        cov = covs_pred

    theta_pred_phys, names_phys = ui.unreparameterize_theta(theta_pred_row, param_names)
    theta_true_phys, _ = ui.unreparameterize_theta(theta_true, param_names)
    names_phys = list(names_phys)

    metrics = {"bx": bx, "n_train": n_train, "n_sims": bx * n_train}
    mse = {}
    mafe = {}
    for key, pname in PARAMS_TRACK.items():
        idx = names_phys.index(pname)
        true_val = float(theta_true_phys[idx])
        pred_val = float(theta_pred_phys[idx])
        err = pred_val - true_val
        mse[key] = err ** 2
        mafe[key] = abs(err / true_val) if true_val != 0 else np.nan
        metrics[f"true_{key}"] = true_val
        metrics[f"pred_{key}"] = pred_val
    metrics["mse"] = mse
    metrics["mafe"] = mafe
    metrics["fom"] = float(ui.figure_of_merit(cov))
    fom_marg, fom_3d = compute_fom_key_block(cov, param_names)
    metrics["fom_marg"] = fom_marg
    metrics["fom_3d"] = fom_3d
    fob, fob3 = compute_fob(theta_pred_row, cov, param_names, theta_true, param_vary)
    metrics["fob"] = fob
    metrics["fob3"] = fob3
    for key in PARAMS_TRACK:
        metrics[f"mse_{key}"] = mse[key]
        metrics[f"mafe_{key}"] = mafe[key]
    return metrics


def _grid_columns():
    cols = ["bx", "n_train", "n_sims", "fom", "fom_3d", "fob3"]
    for key in PARAMS_TRACK:
        cols.extend([f"mse_{key}", f"mafe_{key}", f"true_{key}", f"pred_{key}"])
    for key in FOB_KEYS:
        cols.extend([f"fom_{key}", f"fob_{key}"])
    return cols


def empty_grid_df():
    return pd.DataFrame(columns=_grid_columns())


def collect_grid(
    n_train_values,
    bx_values,
    tag_test=None,
    *,
    statistics=None,
    tag_masks=None,
):
    if tag_test is None:
        tag_test = build_tag_test(statistics=statistics, tag_masks=tag_masks)
    rows = []
    for bx in bx_values:
        for n_train in n_train_values:
            m = load_metrics(
                bx, n_train, tag_test,
                statistics=statistics, tag_masks=tag_masks,
            )
            if m is None:
                print(f"Missing: bx={bx}, n_train={n_train}")
                continue
            row = {
                "bx": bx,
                "n_train": n_train,
                "n_sims": bx * n_train,
                "fom": m["fom"],
                "fom_3d": m["fom_3d"],
                "fob3": m["fob3"],
            }
            for key in PARAMS_TRACK:
                row[f"mse_{key}"] = m["mse"][key]
                row[f"mafe_{key}"] = m["mafe"][key]
                row[f"true_{key}"] = m[f"true_{key}"]
                row[f"pred_{key}"] = m[f"pred_{key}"]
            for key in FOB_KEYS:
                row[f"fom_{key}"] = m["fom_marg"][key]
                row[f"fob_{key}"] = m["fob"][key]
            rows.append(row)
    return pd.DataFrame(rows) if rows else empty_grid_df()


def _apply_logx(ax, logx: bool):
    if logx:
        ax.set_xscale("log")


def _plot_param_panels(df, x_col, xlabel, title_prefix, *, value_prefix, ylabel, logx=False, logy=True):
    if df.empty or x_col not in df.columns:
        print(f"No data for {title_prefix}")
        return
    df = df.sort_values(x_col)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    for ax, key in zip(axes, PARAMS_TRACK):
        ax.plot(df[x_col], df[f"{value_prefix}_{key}"], marker="o", color=PARAM_COLORS[key])
        ax.set_title(PARAM_LABELS[key])
        ax.set_ylabel(ylabel)
        _apply_logx(ax, logx)
        if logy:
            ax.set_yscale("log")
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(f"{title_prefix}: {value_prefix}", y=1.06)
    fig.tight_layout()
    plt.show()


def plot_mse(df, x_col, xlabel, title_prefix, *, logx=False, logy=True):
    _plot_param_panels(
        df, x_col, xlabel, title_prefix,
        value_prefix="mse", ylabel="MSE", logx=logx, logy=logy,
    )


def plot_mafe(df, x_col, xlabel, title_prefix, *, logx=False, logy=True):
    _plot_param_panels(
        df, x_col, xlabel, title_prefix,
        value_prefix="mafe", ylabel="|fractional error|", logx=logx, logy=logy,
    )


def plot_fom(df, x_col, xlabel, title_prefix, *, logx=False, logy=True):
    if df.empty or x_col not in df.columns:
        print(f"No data for {title_prefix} (FoM)")
        return
    df = df.sort_values(x_col)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df[x_col], df["fom"], marker="o", color="k")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("FoM")
    ax.set_title(f"{title_prefix}: figure of merit")
    _apply_logx(ax, logx)
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    plt.show()


def plot_fob(df, x_col, xlabel, title_prefix, *, logx=False):
    if df.empty or x_col not in df.columns:
        print(f"No data for {title_prefix} (FoB)")
        return
    df = df.sort_values(x_col)
    gauss_ref = np.sqrt(2.0 / np.pi)
    lim1 = fob_chi2_sqrt_limit(1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    for ax, key in zip(axes, FOB_KEYS):
        ax.plot(df[x_col], df[f"fob_{key}"], marker="o", color=FOB_COLORS[key])
        ax.axhline(gauss_ref, color="k", ls="--", lw=1, alpha=0.7, label=r"$\sqrt{2/\pi}$")
        ax.axhline(lim1, color="gray", ls=":", lw=1, alpha=0.7, label=rf"$\sqrt{{\chi^2_{{1}}(68\%)}}$")
        ax.set_title(FOB_LABELS[key])
        ax.set_ylabel("FoB")
        _apply_logx(ax, logx)
        if key == FOB_KEYS[0]:
            ax.legend(fontsize=7, loc="best")
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(f"{title_prefix}: FoB $= |\\hat{{\\theta}}-\\theta_{{\\mathrm{{true}}}}|/\\sigma$", y=1.06)
    fig.tight_layout()
    plt.show()


def plot_fob3(df, x_col, xlabel, title_prefix, *, logx=False):
    if df.empty or x_col not in df.columns:
        print(f"No data for {title_prefix} (FoB3D)")
        return
    df = df.sort_values(x_col)
    lim3 = fob_chi2_sqrt_limit(3)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df[x_col], df["fob3"], marker="o", color="k")
    ax.axhline(lim3, color="gray", ls=":", lw=1, alpha=0.7,
               label=rf"$\sqrt{{\chi^2_{{3}}(68\%)}}={lim3:.2f}$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"FoB$_{\mathrm{3D}}$")
    ax.set_title(f"{title_prefix}: FoB$_{{3D}}$ ($P=3$)")
    ax.legend(fontsize=8)
    _apply_logx(ax, logx)
    fig.tight_layout()
    plt.show()


def _sequential_color_map(values, *, cmap_name=BX_CMAP, lo=BX_CMAP_LO, hi=BX_CMAP_HI):
    values_sorted = sorted(values)
    cmap = plt.get_cmap(cmap_name)
    if len(values_sorted) == 1:
        return {values_sorted[0]: cmap(0.5 * (lo + hi))}
    samples = np.linspace(lo, hi, len(values_sorted))
    return {v: cmap(t) for v, t in zip(values_sorted, samples)}


def bx_colors(bx_values, **kwargs):
    return _sequential_color_map(bx_values, **kwargs)


def n_train_colors(n_train_values, **kwargs):
    return _sequential_color_map(n_train_values, **kwargs)


def plot_metric_colored_by_bx(df, x_col, y_col, xlabel, ylabel, title, *, logx=False, logy=False, cmap_name=BX_CMAP):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        print(f"No data for {title}")
        return
    color_map = bx_colors(df["bx"].unique(), cmap_name=cmap_name)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for bx in sorted(df["bx"].unique()):
        sub = df[df["bx"] == bx].sort_values(x_col)
        ax.plot(sub[x_col], sub[y_col], marker="o", ls="-", color=color_map[bx], label=rf"$b_x={bx}$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _apply_logx(ax, logx)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2, title=rf"$b_x$")
    fig.tight_layout()
    plt.show()


def plot_fob_colored_by_bx(df, x_col, xlabel, title_prefix, *, logx=False):
    if df.empty or x_col not in df.columns:
        print(f"No data for {title_prefix} (FoB by $b_x$)")
        return
    color_map = bx_colors(df["bx"].unique())
    gauss_ref = np.sqrt(2.0 / np.pi)
    lim1 = fob_chi2_sqrt_limit(1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    for ax, key in zip(axes, FOB_KEYS):
        for bx in sorted(df["bx"].unique()):
            sub = df[df["bx"] == bx].sort_values(x_col)
            ax.plot(sub[x_col], sub[f"fob_{key}"], marker="o", ls="-", color=color_map[bx], label=rf"$b_x={bx}$")
        ax.axhline(gauss_ref, color="k", ls="--", lw=1, alpha=0.7, label=r"$\sqrt{2/\pi}$")
        ax.axhline(lim1, color="gray", ls=":", lw=1, alpha=0.7, label=rf"$\sqrt{{\chi^2_{{1}}(68\%)}}$")
        ax.set_title(FOB_LABELS[key])
        ax.set_ylabel("FoB")
        _apply_logx(ax, logx)
        if key == FOB_KEYS[0]:
            ax.legend(fontsize=7, loc="best", ncol=2, title=rf"$b_x$")
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(
        f"{title_prefix}: FoB $= |\\hat{{\\theta}}-\\theta_{{\\mathrm{{true}}}}|/\\sigma$ (colored by $b_x$)",
        y=1.08,
    )
    fig.tight_layout()
    plt.show()


def plot_fob3_colored_by_bx(df, x_col, xlabel, title_prefix, *, logx=False):
    if df.empty or x_col not in df.columns:
        print(f"No data for {title_prefix} (FoB3D by $b_x$)")
        return
    color_map = bx_colors(df["bx"].unique())
    lim3 = fob_chi2_sqrt_limit(3)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for bx in sorted(df["bx"].unique()):
        sub = df[df["bx"] == bx].sort_values(x_col)
        ax.plot(sub[x_col], sub["fob3"], marker="o", ls="-", color=color_map[bx], label=rf"$b_x={bx}$")
    ax.axhline(lim3, color="gray", ls=":", lw=1, alpha=0.7, label=rf"$\sqrt{{\chi^2_{{3}}(68\%)}}={lim3:.2f}$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"FoB$_{\mathrm{3D}}$")
    ax.set_title(f"{title_prefix}: FoB$_{{3D}}$ ($P=3$, colored by $b_x$)")
    ax.legend(fontsize=8, ncol=2, title=rf"$b_x$")
    _apply_logx(ax, logx)
    fig.tight_layout()
    plt.show()


def plot_total_sims_colored_by_bx(df, title_prefix):
    xlabel = r"$n_\mathrm{train} \times b_x$"
    plot_metric_colored_by_bx(
        df, "n_sims", "fom", xlabel, "FoM",
        f"{title_prefix}: FoM vs total training sims (colored by $b_x$)",
        logx=True, logy=True,
    )
    plot_fob_colored_by_bx(df, "n_sims", xlabel, title_prefix, logx=True)
    plot_fob3_colored_by_bx(df, "n_sims", xlabel, title_prefix, logx=True)


def plot_all_metrics(df, x_col, xlabel, title_prefix, *, logx=False):
    plot_mse(df, x_col, xlabel, title_prefix, logx=logx)
    plot_mafe(df, x_col, xlabel, title_prefix, logx=logx)
    plot_fob(df, x_col, xlabel, title_prefix, logx=logx)
    plot_fob3(df, x_col, xlabel, title_prefix, logx=logx)
    plot_fom(df, x_col, xlabel, title_prefix, logx=logx)


def _coverage_samples_path(tag_inf: str, tag_test: str):
    base = paths.DIR_RESULTS / "results_sbi" / f"sbi{tag_inf}"
    fn = base / f"samples_test{tag_test}_pred.npy"
    if fn.is_file():
        return fn, False
    fn_ip = base / f"samples_test{tag_test}_pred_inprogress.npy"
    if fn_ip.is_file():
        return fn_ip, True
    return None, False


def _moments_from_samples_3d(samples_arr: np.ndarray, param_names):
    theta_pred = np.mean(samples_arr, axis=0)
    covs_pred = np.array([np.cov(samples_arr[:, i, :].T) for i in range(samples_arr.shape[1])])
    return theta_pred, covs_pred, list(param_names)


def load_coverage_metrics(bx, n_train, tag_test=None, *, statistics=None, tag_masks=None):
    statistics = statistics if statistics is not None else globals()["statistics"]
    tag_masks = tag_masks if tag_masks is not None else globals()["tag_masks"]
    if tag_test is None:
        tag_test = build_tag_test_coverage(statistics=statistics, tag_masks=tag_masks)
    tag_inf = build_tag_inf(bx, n_train, statistics=statistics, tag_masks=tag_masks)
    fn, _ = _coverage_samples_path(tag_inf, tag_test)
    if fn is None:
        return None, "missing"
    samples_arr = np.load(fn)
    fn_pn = paths.DIR_RESULTS / "results_sbi" / f"sbi{tag_inf}" / "param_names.txt"
    if not fn_pn.is_file():
        return None, "missing"
    param_names_all = np.loadtxt(fn_pn, dtype=str)
    if samples_arr.ndim != 3:
        return None, "bad_shape"
    n_obs = int(samples_arr.shape[1])
    if n_obs == 0:
        return None, "missing"
    status = "complete" if n_obs == N_COVERAGE else "incomplete"
    theta_pred, covs_pred, param_names = _moments_from_samples_3d(samples_arr, param_names_all)
    cosmo_vary, bias_vary, param_vary = utils_plot.load_training_params(tag_params, tag_biasparams, bx=bx)
    theta_true_all = data_loader.load_theta_test(
        TAG_PARAMS_TEST_COV, TAG_BIASPARAMS_TEST_COV,
        cosmo_param_names_vary=cosmo_vary, bias_param_names_vary=bias_vary,
    )
    if theta_true_all.ndim == 1:
        theta_true_all = np.tile(theta_true_all, (n_obs, 1))
    n_use = min(n_obs, theta_true_all.shape[0], theta_pred.shape[0])
    theta_true_all = theta_true_all[:n_use]
    theta_pred = theta_pred[:n_use]
    covs_pred = covs_pred[:n_use]
    mse_lists = {key: [] for key in PARAMS_TRACK}
    mafe_lists = {key: [] for key in PARAMS_TRACK}
    fob_lists = {key: [] for key in FOB_KEYS}
    fom_marg_lists = {key: [] for key in FOB_KEYS}
    fob3_list = []
    fom_list = []
    fom_3d_list = []
    for i in range(n_use):
        cov_i = covs_pred[i]
        fom_list.append(float(ui.figure_of_merit(cov_i)))
        fom_marg, fom_3d = compute_fom_key_block(cov_i, param_names)
        fom_3d_list.append(fom_3d)
        for key in FOB_KEYS:
            fom_marg_lists[key].append(fom_marg[key])
        fob, fob3 = compute_fob(theta_pred[i], cov_i, param_names, theta_true_all[i], param_vary)
        for key in FOB_KEYS:
            fob_lists[key].append(fob[key])
        fob3_list.append(fob3)

        theta_pred_phys, names_phys = ui.unreparameterize_theta(theta_pred[i], param_names)
        names_phys = list(names_phys)
        theta_true_phys = {
            pn: float(theta_true_all[i, list(param_vary).index(pn)])
            for pn in param_vary
        }
        for key, pname in PARAMS_TRACK.items():
            true_val = theta_true_phys[pname]
            pred_val = float(theta_pred_phys[names_phys.index(pname)])
            err = pred_val - true_val
            mse_lists[key].append(err ** 2)
            mafe_lists[key].append(abs(err / true_val) if true_val != 0 else np.nan)
    metrics = {
        "n_cov": n_use,
        "fom": float(np.nanmean(fom_list)),
        "fom_3d": float(np.nanmean(fom_3d_list)),
        "fob3": float(np.nanmean(fob3_list)),
        "mse": {key: float(np.nanmean(mse_lists[key])) for key in PARAMS_TRACK},
        "mafe": {key: float(np.nanmean(mafe_lists[key])) for key in PARAMS_TRACK},
        "fom_marg": {key: float(np.nanmean(fom_marg_lists[key])) for key in FOB_KEYS},
        "fob": {key: float(np.nanmean(fob_lists[key])) for key in FOB_KEYS},
    }
    return metrics, status


def collect_coverage_grid(n_train_values, bx_values, tag_test=None, *, statistics=None, tag_masks=None):
    if tag_test is None:
        tag_test = build_tag_test_coverage(statistics=statistics, tag_masks=tag_masks)
    rows = []
    missing = []
    incomplete = []
    for bx in bx_values:
        for n_train in n_train_values:
            m, status = load_coverage_metrics(bx, n_train, tag_test, statistics=statistics, tag_masks=tag_masks)
            if m is None:
                missing.append((bx, n_train, status))
                print(f"Coverage missing ({status}): bx={bx}, n_train={n_train}")
                continue
            if status != "complete":
                incomplete.append((bx, n_train, m["n_cov"], status))
                print(f"Coverage incomplete ({status}): bx={bx}, n_train={n_train}, n_obs={m['n_cov']} / {N_COVERAGE}")
            row = {
                "bx": bx, "n_train": n_train, "n_sims": bx * n_train,
                "n_cov": m["n_cov"], "status": status,
                "fom": m["fom"], "fom_3d": m["fom_3d"], "fob3": m["fob3"],
            }
            for key in PARAMS_TRACK:
                row[f"mse_{key}"] = m["mse"][key]
                row[f"mafe_{key}"] = m["mafe"][key]
            for key in FOB_KEYS:
                row[f"fom_{key}"] = m["fom_marg"][key]
                row[f"fob_{key}"] = m["fob"][key]
            rows.append(row)
    n_expected = len(n_train_values) * len(bx_values)
    print(f"Coverage summary: {len(rows)} loaded, {(pd.DataFrame(rows)['status'] == 'complete').sum() if rows else 0} complete / {n_expected} grid points; {len(missing)} missing, {len(incomplete)} incomplete")
    return pd.DataFrame(rows) if rows else empty_grid_df()


STATISTICS_ARR_FID = [["pk"], ["pk", "pgm"], ["pk", "bispec"], ["pk", "bispec", "pgm"]]
TAG_MASKS_FID = ["", "_kpgm0.25", "_kb0.25", "_kb0.25_kpgm0.25"]


def stat_combo_label(statistics, tag_masks):
    return utils_plot.get_stat_label(statistics) + (f", {tag_masks}" if tag_masks else "")


STAT_COMBO_LABELS = [stat_combo_label(s, m) for s, m in zip(STATISTICS_ARR_FID, TAG_MASKS_FID)]
STAT_COMBO_COLORS = dict(zip(STAT_COMBO_LABELS, utils_plot.get_stat_colors(STATISTICS_ARR_FID)))


def collect_stat_combo_coverage_grids(n_train_values, bx_values, *, complete_only=False):
    frames = []
    for statistics, tag_masks in zip(STATISTICS_ARR_FID, TAG_MASKS_FID):
        label = stat_combo_label(statistics, tag_masks)
        df = collect_coverage_grid(n_train_values, bx_values, statistics=statistics, tag_masks=tag_masks)
        if df.empty:
            print(f"{label}: no coverage data")
            continue
        n_loaded = len(df)
        n_complete = int((df["status"] == "complete").sum())
        print(f"{label}: loaded {n_loaded} grid points ({n_complete} complete, {n_loaded - n_complete} incomplete / {len(n_train_values) * len(bx_values)} expected)")
        if complete_only:
            df = df[df["status"] == "complete"].copy()
            if df.empty:
                print("  (skipping — no complete coverage rows)")
                continue
        df = df.copy()
        df["stat_combo"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else empty_grid_df()


def plot_fom_marg_by_stat_combo(df, x_col, xlabel, title_prefix, *, logx=False):
    if df.empty or x_col not in df.columns:
        print(f"No data for {title_prefix} (marginal FoM)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    for ax, key in zip(axes, FOB_KEYS):
        col = f"fom_{key}"
        for statistics, tag_masks in zip(STATISTICS_ARR_FID, TAG_MASKS_FID):
            label = stat_combo_label(statistics, tag_masks)
            sub = df[df["stat_combo"] == label].sort_values(x_col)
            if sub.empty or col not in sub.columns:
                continue
            color = STAT_COMBO_COLORS[label]
            ax.plot(sub[x_col], sub[col], marker="o", ls="-", color=color, label=label)
            if "status" in sub.columns:
                inc = sub[sub["status"] != "complete"]
                if not inc.empty:
                    ax.plot(inc[x_col], inc[col], marker="o", ls="none", mfc="none", mec=color, mew=1.5)
        ax.set_title(FOB_LABELS[key])
        ax.set_ylabel("FoM")
        ax.set_yscale("log")
        _apply_logx(ax, logx)
        if key == FOB_KEYS[0]:
            ax.legend(fontsize=7, loc="best")
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(f"{title_prefix}: marginal FoM $= 1/\\sigma$ (3-key block)", y=1.06)
    fig.tight_layout()
    plt.show()


def plot_fob_marg_by_stat_combo(df, x_col, xlabel, title_prefix, *, logx=False):
    if df.empty or x_col not in df.columns:
        print(f"No data for {title_prefix} (marginal FoB)")
        return
    gauss_ref = np.sqrt(2.0 / np.pi)
    lim1 = fob_chi2_sqrt_limit(1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    for ax, key in zip(axes, FOB_KEYS):
        col = f"fob_{key}"
        for statistics, tag_masks in zip(STATISTICS_ARR_FID, TAG_MASKS_FID):
            label = stat_combo_label(statistics, tag_masks)
            sub = df[df["stat_combo"] == label].sort_values(x_col)
            if sub.empty or col not in sub.columns:
                continue
            color = STAT_COMBO_COLORS[label]
            ax.plot(sub[x_col], sub[col], marker="o", ls="-", color=color, label=label)
            if "status" in sub.columns:
                inc = sub[sub["status"] != "complete"]
                if not inc.empty:
                    ax.plot(inc[x_col], inc[col], marker="o", ls="none", mfc="none", mec=color, mew=1.5)
        ax.axhline(gauss_ref, color="k", ls="--", lw=1, alpha=0.7, label=r"$\sqrt{2/\pi}$")
        ax.axhline(lim1, color="gray", ls=":", lw=1, alpha=0.7, label=rf"$\sqrt{{\chi^2_{{1}}(68\%)}}$")
        ax.set_title(FOB_LABELS[key])
        ax.set_ylabel("FoB")
        _apply_logx(ax, logx)
        if key == FOB_KEYS[0]:
            ax.legend(fontsize=7, loc="best")
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(
        f"{title_prefix}: FoB $= |\\hat{{\\theta}}-\\theta_{{\\mathrm{{true}}}}|/\\sigma$ (3-key block)",
        y=1.06,
    )
    fig.tight_layout()
    plt.show()


def plot_fom_3d_by_stat_combo(df, x_col, xlabel, title_prefix, *, logx=False):
    plot_metric_by_stat_combo(
        df, x_col, "fom_3d", xlabel, r"FoM$_{3}$",
        f"{title_prefix}: FoM$_{{3}}$ ($P=3$ key block)", logx=logx, logy=True,
    )


def plot_metric_by_stat_combo(df, x_col, y_col, xlabel, ylabel, title, *, logx=False, logy=False, hline=None, hline_label=None, linestyle="-"):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        print(f"No data for {title}")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for statistics, tag_masks in zip(STATISTICS_ARR_FID, TAG_MASKS_FID):
        label = stat_combo_label(statistics, tag_masks)
        sub = df[df["stat_combo"] == label].sort_values(x_col)
        if sub.empty:
            continue
        color = STAT_COMBO_COLORS[label]
        ax.plot(sub[x_col], sub[y_col], ls=linestyle, color=color, label=label)
        if "status" in sub.columns:
            done = sub[sub["status"] == "complete"]
            inc = sub[sub["status"] != "complete"]
            if not done.empty:
                ax.plot(done[x_col], done[y_col], marker="o", ls="none", color=color)
            if not inc.empty:
                ax.plot(inc[x_col], inc[y_col], marker="o", ls="none", mfc="none", mec=color, mew=1.5)
        else:
            ax.plot(sub[x_col], sub[y_col], marker="o", ls="none", color=color)
    if hline is not None:
        ax.axhline(hline, color="gray", ls=":", lw=1, alpha=0.7, label=hline_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _apply_logx(ax, logx)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    plt.show()


NT_ALPHA_LO, NT_ALPHA_HI = 0.15, 1.0


def n_train_alpha(n_train, n_train_values=None, *, lo=NT_ALPHA_LO, hi=NT_ALPHA_HI):
    n_train_values = sorted(n_train_values if n_train_values is not None else n_train_arr)
    if len(n_train_values) == 1:
        return hi
    t = n_train_values.index(int(n_train)) / (len(n_train_values) - 1)
    return lo + t * (hi - lo)


def _stat_combo_nsims_alpha_on_ax(ax, df, y_col, *, n_train_values=None, legend=False):
    n_train_values = sorted(n_train_values if n_train_values is not None else n_train_arr)
    for statistics, tag_masks in zip(STATISTICS_ARR_FID, TAG_MASKS_FID):
        label = stat_combo_label(statistics, tag_masks)
        sub = df[df["stat_combo"] == label]
        if sub.empty or y_col not in sub.columns:
            continue
        rgb = mcolors.to_rgb(STAT_COMBO_COLORS[label])
        if legend:
            ax.plot([], [], color=rgb, marker="o", ls="-", lw=1.5, markersize=6, label=label)
        for n_train in sorted(sub["n_train"].unique()):
            grp = sub[sub["n_train"] == n_train].sort_values("n_sims")
            alpha = n_train_alpha(n_train, n_train_values)
            rgba = (*rgb, alpha)
            if len(grp) >= 2:
                ax.plot(grp["n_sims"], grp[y_col], color=rgba, ls="-", lw=1.5, zorder=1)
            if "status" in grp.columns:
                done = grp[grp["status"] == "complete"]
                inc = grp[grp["status"] != "complete"]
            else:
                done, inc = grp, grp.iloc[0:0]
            if not done.empty:
                ax.scatter(done["n_sims"], done[y_col], c=[rgba] * len(done), marker="o", s=36, linewidths=0, zorder=2)
            for _, row in inc.iterrows():
                ax.scatter(row["n_sims"], row[y_col], facecolors="none", edgecolors=rgba, marker="o", s=36, linewidths=1.5, zorder=2)


def plot_fom_marg_by_stat_combo_nsims_alpha(df, title_prefix, *, logx=True, n_train_values=None):
    if df.empty or "n_sims" not in df.columns:
        print(f"No data for {title_prefix} (marginal FoM vs n_sims)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    for ax, key in zip(axes, FOB_KEYS):
        col = f"fom_{key}"
        _stat_combo_nsims_alpha_on_ax(ax, df, col, n_train_values=n_train_values, legend=(key == FOB_KEYS[0]))
        ax.set_title(FOB_LABELS[key])
        ax.set_ylabel("FoM")
        ax.set_yscale("log")
        _apply_logx(ax, logx)
        if key == FOB_KEYS[0]:
            ax.legend(fontsize=7, loc="best", title="stat combo")
    axes[-1].set_xlabel(r"$n_\mathrm{train} \times b_x$")
    fig.suptitle(f"{title_prefix}: marginal FoM vs total training sims", y=1.06)
    axes[0].text(0.02, 0.02, r"opacity $\propto n_\mathrm{train}$", transform=axes[0].transAxes, fontsize=8, color="0.35", va="bottom")
    fig.tight_layout()
    plt.show()


def plot_fob_marg_by_stat_combo_nsims_alpha(df, title_prefix, *, logx=True, n_train_values=None):
    if df.empty or "n_sims" not in df.columns:
        print(f"No data for {title_prefix} (marginal FoB vs n_sims)")
        return
    gauss_ref = np.sqrt(2.0 / np.pi)
    lim1 = fob_chi2_sqrt_limit(1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    for ax, key in zip(axes, FOB_KEYS):
        col = f"fob_{key}"
        _stat_combo_nsims_alpha_on_ax(ax, df, col, n_train_values=n_train_values, legend=(key == FOB_KEYS[0]))
        ax.axhline(gauss_ref, color="k", ls="--", lw=1, alpha=0.7)
        ax.axhline(lim1, color="gray", ls=":", lw=1, alpha=0.7)
        ax.set_title(FOB_LABELS[key])
        ax.set_ylabel("FoB")
        _apply_logx(ax, logx)
        if key == FOB_KEYS[0]:
            ax.legend(fontsize=7, loc="best", title="stat combo")
    axes[-1].set_xlabel(r"$n_\mathrm{train} \times b_x$")
    fig.suptitle(f"{title_prefix}: marginal FoB vs total training sims", y=1.06)
    axes[0].text(0.02, 0.02, r"opacity $\propto n_\mathrm{train}$", transform=axes[0].transAxes, fontsize=8, color="0.35", va="bottom")
    fig.tight_layout()
    plt.show()


def plot_metric_by_stat_combo_nsims_alpha(df, y_col, ylabel, title, *, logx=True, logy=False, hline=None, hline_label=None, n_train_values=None):
    if df.empty or "n_sims" not in df.columns or y_col not in df.columns:
        print(f"No data for {title}")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _stat_combo_nsims_alpha_on_ax(ax, df, y_col, n_train_values=n_train_values, legend=True)
    if hline is not None:
        ax.axhline(hline, color="gray", ls=":", lw=1, alpha=0.7, label=hline_label)
    ax.set_xlabel(r"$n_\mathrm{train} \times b_x$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _apply_logx(ax, logx)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best", title="stat combo")
    ax.text(0.02, 0.02, r"opacity $\propto n_\mathrm{train}$", transform=ax.transAxes, fontsize=8, color="0.35", va="bottom")
    fig.tight_layout()
    plt.show()
