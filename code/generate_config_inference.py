import os
import yaml
from pathlib import Path

import paths

'''
Generates YAML configuration files for inference.
'''

# Fiducial bx / n_train used to name the wandb sweep (sweep_name) and for
# run_mode `best`: if (bx, n_train) in the config match these, the best run
# artifact is copied; otherwise best hyperparameters are taken from the sweep
# and the model is retrained on the config's bx / n_train.
# Training data always uses config ``bx`` and ``n_train`` (including run_mode
# ``sweep``); only hyperparameters come from the wandb sweep.
BX_SWEEP = 32
N_TRAIN_SWEEP = 10000
SWEEP_NUM_RUNS = 16
TAG_SWEEP = f"-rand{SWEEP_NUM_RUNS}"  # sweep/best dir suffix; single source of truth
# Scale cuts (labels from data_loader.K_CUTS; a cut keeps every bin with top edge <= label).
# SWEEP_MASKS: where hyperparameter sweeps are run (and named); models at other cuts reuse
# that sweep's hyperparameters. FIDUCIAL_MASKS: the analysis fiducial (currently the same).
SWEEP_MASKS = {"pk": "_kp0.37", "bispec": "_kb0.24", "pgm": "_kpgm0.24"}
FIDUCIAL_MASKS = dict(SWEEP_MASKS)
# v2 test cosmology/IC sets (own IC seeds; v1 _coverage/_shame shared phases with training).
TAG_PARAMS_COVERAGE = "_coveragev2_p5_n1000"
TAG_PARAMS_SHAME = "_shamev2_p0_n1000"
NBAR_TAGS_SHAME = ["_nbar0.00011", "_nbar0.00022", "_nbar0.00054"]  # SHAMe OOD mocks
TAG_MOCK_SHAME_FIDUCIAL = "_nbar0.00022"


def tags_mask_for_sweep(statistics):
    """Per-statistic masks (``SWEEP_MASKS``) at which a statistic combo's sweep is run/named."""
    return [SWEEP_MASKS[s] for s in statistics]

# Resolve config output directories relative to repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS_TRAIN_DIR = REPO_ROOT / "configs" / "configs_train"
DEFAULT_CONFIGS_TEST_DIR = REPO_ROOT / "configs" / "configs_test"

NOISE_MODES = ("noiseless", "noisy")
NOISE_MODE_TRAIN_BIAS = {
    "noiseless": "_biasnest_p4_n320000",
    "noisy": "_biasnoisenestv2_p9_n320000",
}


def resolve_train_tag_bundle(tag_params: str, noise_mode: str) -> dict:
    """
    From cosmo ``tag_params`` and ``noise_mode`` (``noiseless`` or ``noisy``),
    return ``tag_params``, ``tag_biasparams``, and ``tag_noise`` for training configs.
    """
    if noise_mode not in NOISE_MODE_TRAIN_BIAS:
        raise KeyError(
            f"Unknown noise_mode {noise_mode!r}; expected one of {list(NOISE_MODE_TRAIN_BIAS)}"
        )
    out = {
        "tag_params": tag_params,
        "tag_biasparams": NOISE_MODE_TRAIN_BIAS[noise_mode],
        "tag_noise": None if noise_mode == "noiseless" else "_noisefly",
    }
    return out


def build_tag_data(
    data_mode: str,
    statistics: list[str],
    tags_mask: list[str],
    tag_params: str,
    tag_biasparams: str,
    tag_noise: str | None = None,
) -> str:
    """Build ``tag_data`` string (leading ``_muchisimocks`` segment + stats + masks + param tags)."""
    tag_stats = f'_{"_".join(statistics)}'
    tag_masks = "".join(tags_mask)
    tag_paramsall = tag_params + tag_biasparams
    if tag_noise is not None:
        tag_paramsall += tag_noise
    return f"_{data_mode}{tag_stats}{tag_masks}{tag_paramsall}"


def build_sweep_name(data_mode, statistics, tag_params, tag_biasparams, tag_noise,
                     reparameterize, tag_sweep, tags_mask=None) -> str:
    """Sweep directory tag (``results_sbi/sbi<name>``) for a training bundle.

    The one place sweep names are built; ``tags_mask`` defaults to the fiducial
    ``tags_mask_for_sweep(statistics)``, bx/n_train are always BX_SWEEP/N_TRAIN_SWEEP.
    """
    if tags_mask is None:
        tags_mask = tags_mask_for_sweep(statistics)
    tag_data = build_tag_data(data_mode, statistics, tags_mask, tag_params, tag_biasparams, tag_noise)
    return (tag_data + ("_rp" if reparameterize else "")
            + f"_bx{BX_SWEEP}_ntrain{N_TRAIN_SWEEP}_sweep{tag_sweep}")


def nth_passing_run_id(sweep_dir: Path, nth: int = 0) -> str:
    """Line ``nth`` of ``best_runs.txt`` (0-based). ``nth=0`` may fall back to ``best_run.txt``."""
    runs_path = sweep_dir / "best_runs.txt"
    if runs_path.is_file():
        ids = [
            ln.strip()
            for ln in runs_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if nth < len(ids):
            return ids[nth]
        raise IndexError(
            "nth_best_run=%d but %s has only %d id(s). "
            "Re-run choose_best_run.py with --n-passing at least %d."
            % (nth, runs_path, len(ids), nth + 1)
        )
    if nth == 0:
        id_path = sweep_dir / "best_run.txt"
        if id_path.is_file():
            rid = id_path.read_text(encoding="utf-8").split()[0].strip()
            if rid:
                return rid
    raise FileNotFoundError(
        "nth_best_run=%d needs %s with >= %d lines (or run choose_best_run.py). Path: %s"
        % (nth, runs_path.name, nth + 1, sweep_dir)
    )


def best_tag(tag_sweep: str, nth_best_run: int | None) -> str:
    """``_best{tag_sweep}`` (top run, ``nth_best_run`` None or 0) or ``..._nbest{n}`` (n-th run)."""
    nth = 0 if nth_best_run is None else int(nth_best_run)
    return f"_best{tag_sweep}" + (f"_nbest{nth}" if nth else "")


def _tag_inf_train_for_test(
    base_inf_train: str,
    tag_sweep: str | None,
    nth_best_run: int | None,
    seed: int = 0,
) -> str:
    """``tag_inf`` of the trained model a test config evaluates (= generate_train_config's).

    Best-mode models always live in their own ``_best...`` dir (at the sweep's setup the
    best-mode training step copies the sweep trial there), never in the sweep trial dir.
    """
    if tag_sweep is None:
        return base_inf_train + _seed_suffix(seed)
    return base_inf_train + best_tag(tag_sweep, nth_best_run) + _seed_suffix(seed)


def _seed_suffix(seed: int) -> str:
    """Tag suffix for a retrain with a non-default training seed (seed 0 = the sweep's seed)."""
    return f"_seed{int(seed)}" if int(seed) != 0 else ""


def build_config_tag_test(
    tag_inf_train: str,
    tag_data_test: str,
    data_mode: str,
    data_mode_test: str,
    statistics: list[str],
    tags_mask: list[str],
    evaluate_mean: bool = False,
    idxs_obs_subset: str | None = None,
) -> str:
    """
    Tag for test config filenames: ``config{tag}.yaml``.

    Train and test usually share ``data_mode``, statistics, and k-bin masks; those
    appear already in ``tag_inf_train``, so omit the repeated prefix from the
    ``_TEST`` segment. ``tag_data_test`` in the YAML is unchanged.

    ``idxs_obs_subset`` (named subset of test rows) is appended, as run_inference does for the
    samples file.
    """
    tag_mean = "_mean" if evaluate_mean else ""
    tag_mean += f"_{idxs_obs_subset}" if idxs_obs_subset else ""
    tag_inf_file = tag_inf_train
    tag_stats = f'_{"_".join(statistics)}'
    tag_masks = "".join(tags_mask)
    shared_prefix = f"_{data_mode}{tag_stats}{tag_masks}"
    if data_mode_test == data_mode and tag_data_test.startswith(shared_prefix):
        test_part = tag_data_test[len(shared_prefix):]
        return f"_TRAIN{tag_inf_file}_TEST{test_part}{tag_mean}"
    return f"_TRAIN{tag_inf_file}_TEST{tag_data_test}{tag_mean}"


# Default cosmo tag only; prefer ``resolve_train_tag_bundle(tag_params, noise_mode)``.
PARAM_SETS_TRAIN = {
    mode: resolve_train_tag_bundle("_p5_n10000", mode) for mode in NOISE_MODES
}


# Test scenario cores: ``tag_biasparams_test`` / ``tag_noise_test`` come from
# ``resolve_test_scenario_tags`` using ``noise_mode`` and ``tag_params_test``.
PARAM_SETS_TEST = {
    "coverage": dict(
        evaluate_mean=False,
        data_mode_test="muchisimocks",
        tag_params_test=TAG_PARAMS_COVERAGE,
    ),
    "fixed_cosmo_shame_mean": dict(
        evaluate_mean=True,
        data_mode_test="muchisimocks",
        tag_params_test=TAG_PARAMS_SHAME,
    ),
    "fixed_cosmo_shame_sample": dict(
        evaluate_mean=False,
        data_mode_test="muchisimocks",
        tag_params_test=TAG_PARAMS_SHAME,
    ),
    "ood": dict(
        _ood=True,
        evaluate_mean=False,
        data_mode_test="shame",
        tag_mock=TAG_MOCK_SHAME_FIDUCIAL,  # override with tag_mock=...; main() writes all NBAR_TAGS_SHAME
        idxs_obs=None,
    ),
}

# In-distribution test presets only. Values are (tag_biasparams_test, tag_noise_test) where
# tag_noise_test None means noiseless; "_noisefly" means on-the-fly per-row unit noise.
# For shame + noisy, bias stays ``_biasshame_*`` (fixed bias draw); only the noise-field tag is derived.
_TEST_SCENARIO_TAGS = {
    "coverage": {
        "noiseless": ("_biascoverage_p4_n1000", None),
        "noisy": ("_biasnoisecoveragev2_p9_n1000", "_noisefly"),
    },
    "fixed_cosmo_shame_mean": {
        "noiseless": ("_biasshame_p0_n1", None),
        "noisy": ("_biasshame_noisebest_p0_n1", "_noisefly"),
    },
    "fixed_cosmo_shame_sample": {
        "noiseless": ("_biasshame_p0_n1", None),
        "noisy": ("_biasshame_noisebest_p0_n1", "_noisefly"),
    },
}


def resolve_test_scenario_tags(scenario, noise_mode, tag_params_test):
    """Return ``tag_biasparams_test`` and ``tag_noise_test`` for an in-distribution test preset."""
    if scenario not in _TEST_SCENARIO_TAGS:
        return {}
    if noise_mode not in NOISE_MODES:
        raise KeyError(f"Unknown noise_mode {noise_mode!r}; expected one of {NOISE_MODES}")
    bias_t, noise_t = _TEST_SCENARIO_TAGS[scenario][noise_mode]
    return {"tag_biasparams_test": bias_t, "tag_noise_test": noise_t}


def generate_train_config(dir_config=str(DEFAULT_CONFIGS_TRAIN_DIR),
                          overwrite=False,
                          statistics=['pk'],
                          n_train=N_TRAIN_SWEEP,
                          bx=BX_SWEEP,
                          data_mode="muchisimocks",
                          tag_params="_p5_n10000",
                          tag_biasparams="_biasnest_p4_n320000",
                          tag_noise=None,
                          tags_mask=None,
                          reparameterize=True,
                          run_mode="single",
                          tag_sweep=None,
                          nth_best_run: int | None = None,
                          seed: int = 0):
    """
    Generates a YAML configuration file for training.

    For ``run_mode`` ``sweep``, ``tag_inf`` / ``sweep_name`` use the training
    ``tags_mask`` (and this config's bx / n_train) so two k-cuts do not share a
    sweep directory. ``wandb_sweep_id`` (initially ``None``) and ``sweep_num_runs``
    are written so re-running this same config can resume without env vars.

    For ``run_mode`` ``best``, ``tag_sweep`` must match the sweep you optimized;
    ``sweep_name`` is the fiducial-mask sweep (``build_sweep_name``), not the training
    ``tags_mask``.

    ``nth_best_run``: if set, training uses line ``n`` of the sweep's ``best_runs.txt`` (not
    ``best_run.txt``) and ``tag_inf`` is ``_best{tag_sweep}_nbest{n}`` (no suffix for n=0).

    ``seed``: training seed; non-zero appends ``_seed{seed}`` to ``tag_inf`` (and always retrains
    in best mode, since sweep trials use seed 0).
    """
    assert not (run_mode == "sweep" and int(seed) != 0), "sweep trials use seed 0"
    # bx is bias parameters per cosmo (1x, 2x, 4x, 8x, 16x, 32x)
    tags_mask = [""] * len(statistics) if tags_mask is None else list(tags_mask)
    assert len(tags_mask) == len(statistics), (
        f"tags_mask length ({len(tags_mask)}) must match statistics length ({len(statistics)}): "
        f"tags_mask={tags_mask!r}, statistics={statistics!r}"
    )

    tag_data_train = build_tag_data(
        data_mode, statistics, tags_mask, tag_params, tag_biasparams, tag_noise
    )

    tag_inf_num = f'_bx{bx}_ntrain{n_train}'
    base_inf = tag_data_train + ('_rp' if reparameterize else '') + tag_inf_num

    if run_mode == 'sweep':
        # Named after the data actually trained on (normally the fiducial mask).
        assert (bx, n_train) == (BX_SWEEP, N_TRAIN_SWEEP), (
            f"sweeps are named with bx{BX_SWEEP}/ntrain{N_TRAIN_SWEEP}; got bx{bx}/ntrain{n_train}")
        sweep_name = build_sweep_name(data_mode, statistics, tag_params, tag_biasparams, tag_noise,
                                      reparameterize, tag_sweep, tags_mask=tags_mask)
        tag_inf = sweep_name
    elif run_mode == 'best':
        # Output dir tag includes this bx/n_train; sweep_name is the completed fiducial sweep.
        tag_inf = base_inf + best_tag(tag_sweep, nth_best_run) + _seed_suffix(seed)
        sweep_name = build_sweep_name(data_mode, statistics, tag_params, tag_biasparams, tag_noise,
                                      reparameterize, tag_sweep)
    else:  # single
        tag_inf = base_inf + _seed_suffix(seed)
        sweep_name = None

    config = {
        "data_mode": data_mode,
        "statistics": statistics,
        "tag_params": tag_params,
        "tag_biasparams": tag_biasparams,
        "tag_noise": tag_noise,
        "tags_mask": tags_mask,
        "n_train": n_train,
        "bx": bx,
        "run_mode": run_mode,
        "tag_sweep": tag_sweep,
        "sweep_name": sweep_name,
        "tag_data": tag_data_train,
        "tag_inf": tag_inf,
        "reparameterize": reparameterize,
    }
    if nth_best_run is not None:
        config["nth_best_run"] = int(nth_best_run)
    if int(seed) != 0:
        config["seed"] = int(seed)
    if run_mode == "sweep":
        config["wandb_sweep_id"] = None
        config["sweep_num_runs"] = SWEEP_NUM_RUNS
            
    os.makedirs(dir_config, exist_ok=True)
    fn_config = f"{dir_config}/config{tag_inf}.yaml"
    if not overwrite and os.path.exists(fn_config):
        print(f"Config file already exists: {fn_config}")
        print("Set overwrite=True to overwrite the existing file.")
        return fn_config
    else:
        if os.path.exists(fn_config):
            print("Config file already exists but overwrite=True, overwriting. Hope you meant to do that!")
        with open(fn_config, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        print(f"Training config file written: {fn_config}")
        return fn_config


def generate_test_config(dir_config=str(DEFAULT_CONFIGS_TEST_DIR),
                         overwrite=False,
                         statistics=['pk'],
                         n_train=N_TRAIN_SWEEP,
                         bx=BX_SWEEP,
                         data_mode="muchisimocks",
                         tag_params="_p5_n10000",
                         tag_biasparams="_biasnest_p4_n320000",
                         tag_noise=None,
                         tags_mask=None,
                         reparameterize=True,
                         tag_sweep=None,
                         nth_best_run: int | None = None,
                         data_mode_test="muchisimocks",
                         idxs_obs=None,
                         idxs_obs_subset: str | None = None,
                         evaluate_mean=True,
                         tag_params_test=TAG_PARAMS_SHAME,
                         tag_biasparams_test="_biasshame_p0_n1",
                         tag_noise_test=None,
                         seed: int = 0):
    """
    Generates a YAML configuration file for testing.

    ``tags_mask`` is a list (length ``len(statistics)``) of per-statistic suffix strings
    (e.g. ``""`` or ``"_kb0.24"`` for bispec); joined into ``tag_masks`` for
    ``tag_data_train`` / ``tag_data_test`` / ``tag_inf_train``, matching training configs.

    If ``tag_sweep`` is set, tests the best-mode model ``_best<tag_sweep>[_nbest<n>][_seed<s>]``
    trained at this config's ``tags_mask`` (``nth_best_run`` None/0 = top run).
    ``idxs_obs_subset``: named row subset, ``data/params/idxs_obs{tag_params_test}_{name}.npy``
    (code_auxiliary/select_coverage_subset.py); appended to the config filename.
    """
    tag_stats = f'_{"_".join(statistics)}'

    tags_mask = [""] * len(statistics) if tags_mask is None else list(tags_mask)
    assert len(tags_mask) == len(statistics), (
        f"tags_mask length ({len(tags_mask)}) must match statistics length ({len(statistics)}): "
        f"tags_mask={tags_mask!r}, statistics={statistics!r}"
    )
    tag_masks = "".join(tags_mask)

    tag_data_train = build_tag_data(
        data_mode, statistics, tags_mask, tag_params, tag_biasparams, tag_noise
    )
    tag_paramsall_test = tag_params_test + tag_biasparams_test
    if tag_noise_test is not None:
        tag_paramsall_test += tag_noise_test
    tag_data_test = f"_{data_mode_test}{tag_stats}{tag_masks}{tag_paramsall_test}"

    tag_inf_num = f"_bx{bx}_ntrain{n_train}"
    base_inf_train = tag_data_train + ("_rp" if reparameterize else "") + tag_inf_num
    sweep_name = (
        None if tag_sweep is None else
        build_sweep_name(data_mode, statistics, tag_params, tag_biasparams, tag_noise,
                         reparameterize, tag_sweep)
    )
    tag_inf_train = _tag_inf_train_for_test(base_inf_train, tag_sweep, nth_best_run, seed=seed)

    tag_test = build_config_tag_test(
        tag_inf_train,
        tag_data_test,
        data_mode,
        data_mode_test,
        statistics,
        tags_mask,
        evaluate_mean=evaluate_mean,
        idxs_obs_subset=idxs_obs_subset,
    )
    
    config = {
        "data_mode": data_mode,
        "data_mode_test": data_mode_test,
        "statistics": statistics,
        "tag_params": tag_params,
        "tag_biasparams": tag_biasparams,
        "tag_noise": tag_noise,
        "tags_mask": tags_mask,
        "tag_params_test": tag_params_test,
        "tag_biasparams_test": tag_biasparams_test,
        "tag_noise_test": tag_noise_test,
        "n_train": n_train,
        "bx": bx,
        "tag_sweep": tag_sweep,
        "nth_best_run": nth_best_run,
        "idxs_obs_subset": idxs_obs_subset,
        "seed": int(seed),
        "evaluate_mean": evaluate_mean,
        "idxs_obs": idxs_obs,
        "tag_data_train": tag_data_train,
        "tag_inf_train": tag_inf_train,
        "sweep_name": sweep_name,
        "tag_data_test": tag_data_test,
        "tag_test": tag_test,
        "reparameterize": reparameterize,
    }
    
    os.makedirs(dir_config, exist_ok=True)
    fn_config = f"{dir_config}/config{tag_test}.yaml"
    if not overwrite and os.path.exists(fn_config):
        print(f"Config file already exists: {fn_config}")
        print("Set overwrite=True to overwrite the existing file.")
        return fn_config
    else:
        if os.path.exists(fn_config):
            print("Config file already exists but overwrite=True, overwriting. Hope you meant to do that!")
        with open(fn_config, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        print(f"Testing config file written: {fn_config}")
        return fn_config


def generate_test_config_ood(dir_config=str(DEFAULT_CONFIGS_TEST_DIR),
                             overwrite=False,
                             statistics=['pk'],
                             n_train=N_TRAIN_SWEEP,
                             bx=BX_SWEEP,
                             data_mode="muchisimocks",
                             tag_params="_p5_n10000",
                             tag_biasparams="_biasnest_p4_n320000",
                             tag_noise=None,
                             tags_mask=None,
                             reparameterize=True,
                             tag_sweep=None,
                             nth_best_run: int | None = None,
                             idxs_obs=None,
                             evaluate_mean=False,
                             data_mode_test="shame",
                             tag_mock=TAG_MOCK_SHAME_FIDUCIAL,
                             seed: int = 0):
    """
    Generates a YAML configuration file for OOD testing.
    """
    tag_stats = f'_{"_".join(statistics)}'
    tags_mask = [""] * len(statistics) if tags_mask is None else list(tags_mask)
    assert len(tags_mask) == len(statistics), (
        f"tags_mask length ({len(tags_mask)}) must match statistics length ({len(statistics)}): "
        f"tags_mask={tags_mask!r}, statistics={statistics!r}"
    )
    tag_masks_train = "".join(tags_mask)

    tag_data_train = build_tag_data(
        data_mode, statistics, tags_mask, tag_params, tag_biasparams, tag_noise
    )
    tag_inf_num = f"_bx{bx}_ntrain{n_train}"
    base_inf_train = tag_data_train + ("_rp" if reparameterize else "") + tag_inf_num
    sweep_name = (
        None if tag_sweep is None else
        build_sweep_name(data_mode, statistics, tag_params, tag_biasparams, tag_noise,
                         reparameterize, tag_sweep)
    )
    tag_inf_train = _tag_inf_train_for_test(base_inf_train, tag_sweep, nth_best_run, seed=seed)

    ### test tags
    tag_data_test = '_'+data_mode_test + tag_stats + tag_masks_train + tag_mock
    
    tag_test = build_config_tag_test(
        tag_inf_train,
        tag_data_test,
        data_mode,
        data_mode_test,
        statistics,
        tags_mask,
        evaluate_mean=evaluate_mean,
    )
    
    config = {
        "data_mode": data_mode,
        "data_mode_test": data_mode_test,
        "statistics": statistics,
        "tag_params": tag_params,
        "tag_biasparams": tag_biasparams,
        "tag_noise": tag_noise,
        "tags_mask": tags_mask,
        "n_train": n_train,
        "bx": bx,
        "tag_sweep": tag_sweep,
        "nth_best_run": nth_best_run,
        "seed": int(seed),
        "evaluate_mean": evaluate_mean,
        "idxs_obs": idxs_obs,
        "tag_data_train": tag_data_train,
        "tag_inf_train": tag_inf_train,
        "sweep_name": sweep_name,
        "tag_data_test": tag_data_test,
        "tag_test": tag_test,
        "tag_mock": tag_mock,
        "reparameterize": reparameterize,
    }
    
    os.makedirs(dir_config, exist_ok=True)
    fn_config = f"{dir_config}/config{tag_test}.yaml"
    if not overwrite and os.path.exists(fn_config):
        print(f"Config file already exists: {fn_config}")
        print("Set overwrite=True to overwrite the existing file.")
        return fn_config
    else:
        if os.path.exists(fn_config):
            print("Config file already exists but overwrite=True, overwriting. Hope you meant to do that!")
        with open(fn_config, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        print(f"Testing config file written: {fn_config}")
        return fn_config


def generate_test_config_from_preset(
    preset_name,
    *,
    tag_params="_p5_n10000",
    noise_mode="noiseless",
    dir_config=str(DEFAULT_CONFIGS_TEST_DIR),
    overwrite=False,
    statistics=['pk'],
    n_train=N_TRAIN_SWEEP,
    bx=BX_SWEEP,
    tags_mask=None,
    tag_sweep=None,
    **kwargs,
):
    """
    Build a test or OOD config from ``PARAM_SETS_TEST``, ``tag_params``, and ``noise_mode``.

    Train-side tags use ``resolve_train_tag_bundle(tag_params, noise_mode)``. In-distribution
    test presets also get ``tag_biasparams_test`` / ``tag_noise_test`` from
    ``resolve_test_scenario_tags`` (OOD presets only use train tags).

    ``tags_mask`` must align with ``statistics`` (length and order): same list as in
    ``generate_train_config`` / ``generate_test_config`` so ``tag_data_train``,
    ``tag_inf_train``, and (in-distribution) ``tag_data_test`` include the correct
    ``tag_masks`` segment (e.g. bispec k-bin masks).

    Pass ``tag_sweep`` (e.g. ``TAG_SWEEP``) and optionally ``nth_best_run`` / ``seed`` to test a
    best-mode model; omit ``tag_sweep`` for a non-sweep training run. Extra kwargs (e.g.
    ``tag_mock``, ``idxs_obs_subset``) override the preset.
    """
    if preset_name not in PARAM_SETS_TEST:
        raise KeyError(
            f"Unknown test preset {preset_name!r}; choose from {list(PARAM_SETS_TEST)}"
        )
    preset = PARAM_SETS_TEST[preset_name]
    ood = preset.get("_ood", False)
    test_fields = {k: v for k, v in preset.items() if k != "_ood"}
    train_fields = resolve_train_tag_bundle(tag_params, noise_mode)
    tag_params_test = preset.get("tag_params_test")
    scenario_tags = resolve_test_scenario_tags(preset_name, noise_mode, tag_params_test)
    common = {
        **test_fields,
        "dir_config": dir_config,
        "overwrite": overwrite,
        "statistics": statistics,
        "n_train": n_train,
        "bx": bx,
        "tags_mask": tags_mask,
        "tag_sweep": tag_sweep,
        **train_fields,
        **scenario_tags,
        **kwargs,
    }
    if ood:
        return generate_test_config_ood(**common)
    return generate_test_config(**common)


STAT_COMBOS = (["pk"], ["pk", "pgm"], ["pk", "bispec"], ["pk", "bispec", "pgm"])


def main():
    """Write the configs for one step of the v2 pipeline; edit the settings below.

    Default: fiducial-cut sweep configs for every statistic combo. Scale-cut / convergence
    grids are built from ensemble_kmax_utils (``iter_*_configs``) in their job scripts.
    """
    overwrite = False
    noise_mode = "noisy"  # see NOISE_MODE_TRAIN_BIAS
    tag_params_train = "_p5_n10000"
    mode = "train"  # "train" | "test"
    run_mode = "sweep"  # train only: "sweep" | "best"
    nth_best_run = None  # best / test: line of the sweep's best_runs.txt (None = top run)
    seed = 0  # best / test: training seed (0 = the sweep's seed)
    test_names = ["coverage", "fixed_cosmo_shame_mean", "ood"]
    coverage_subset = None  # e.g. "inner500_rand100" (select_coverage_subset.py); None = all rows
    n_train, bx = N_TRAIN_SWEEP, BX_SWEEP

    train_kw = resolve_train_tag_bundle(tag_params_train, noise_mode)
    for statistics in STAT_COMBOS:
        masks = SWEEP_MASKS if (mode == "train" and run_mode == "sweep") else FIDUCIAL_MASKS
        tags_mask = [masks[s] for s in statistics]
        if mode == "train":
            generate_train_config(
                overwrite=overwrite, statistics=list(statistics), tags_mask=tags_mask,
                n_train=n_train, bx=bx, run_mode=run_mode, tag_sweep=TAG_SWEEP,
                nth_best_run=nth_best_run, seed=seed, **train_kw,
            )
        elif mode == "test":
            for test_name in test_names:
                common = dict(
                    tag_params=tag_params_train, noise_mode=noise_mode,
                    overwrite=overwrite, statistics=list(statistics), n_train=n_train, bx=bx,
                    tags_mask=tags_mask, tag_sweep=TAG_SWEEP, nth_best_run=nth_best_run,
                    seed=seed,
                )
                if test_name == "ood":
                    for tag_mock in NBAR_TAGS_SHAME:
                        generate_test_config_from_preset(test_name, tag_mock=tag_mock, **common)
                elif test_name == "coverage":
                    generate_test_config_from_preset(
                        test_name, idxs_obs_subset=coverage_subset, **common)
                else:
                    generate_test_config_from_preset(test_name, **common)
        else:
            raise ValueError(f"mode={mode!r}")


if __name__ == "__main__":
    main()
