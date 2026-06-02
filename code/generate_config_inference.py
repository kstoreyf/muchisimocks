import os
import yaml
from pathlib import Path

import utils_model

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
SWEEP_NUM_RUNS = 30
# Fiducial bispec k-bin suffix in sweep / best_run tag strings (like BX_SWEEP for bx).
TAG_MASK_BISPEC_SWEEP = "_kb0.25"


def tags_mask_for_sweep(statistics):
    """Per-statistic masks for ``sweep_name`` / ``base_inf_sweep`` only: bispec → ``TAG_MASK_BISPEC_SWEEP``, else ``''``."""
    return [TAG_MASK_BISPEC_SWEEP if s == "bispec" else "" for s in statistics]

# Resolve config output directories relative to repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS_TRAIN_DIR = REPO_ROOT / "configs" / "configs_train"
DEFAULT_CONFIGS_TEST_DIR = REPO_ROOT / "configs" / "configs_test"
DEFAULT_CONFIGS_RUNLIKE_DIR = REPO_ROOT / "configs" / "configs_runlike"

NOISE_MODES = ("noiseless", "noisy", "noisym2")

# Nested train bias tags (cosmo LH tag is ``tag_params``).
# noisym2: second-order multiplicative noise (m2p3), see generate_params biasnoisem2nest_*.
NOISE_MODE_TRAIN_BIAS = {
    "noiseless": "_biasnest_p4_n320000",
    "noisy": "_biasnoisenest_p9_n320000",
    "noisym2": "_biasnoisem2nest_p7_n320000",
}


def resolve_train_tag_bundle(tag_params: str, noise_mode: str) -> dict:
    """
    From cosmo ``tag_params`` and ``noise_mode`` (``noiseless``, ``noisy``, or ``noisym2``),
    return ``tag_params``, ``tag_biasparams``, and ``tag_noise`` for training configs.
    """
    if noise_mode not in NOISE_MODE_TRAIN_BIAS:
        raise KeyError(
            f"Unknown noise_mode {noise_mode!r}; expected one of {list(NOISE_MODE_TRAIN_BIAS)}"
        )
    if noise_mode != "noiseless" and not tag_params.startswith("_"):
        raise ValueError(f"tag_params must start with '_', got {tag_params!r}")
    out = {
        "tag_params": tag_params,
        "tag_biasparams": NOISE_MODE_TRAIN_BIAS[noise_mode],
        "tag_noise": None if noise_mode == "noiseless" else "_noise_unit" + tag_params,
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
        tag_params_test="_coverage_p5_n1000",
    ),
    "fixed_cosmo_shame_mean": dict(
        evaluate_mean=True,
        data_mode_test="muchisimocks",
        tag_params_test="_shame_p0_n1000",
    ),
    "fixed_cosmo_shame_sample": dict(
        evaluate_mean=False,
        data_mode_test="muchisimocks",
        tag_params_test="_shame_p0_n1000",
    ),
    "ood": dict(
        _ood=True,
        evaluate_mean=False,
        data_mode_test="shame",
        #tag_mock="_nbar0.00011",
        tag_mock="_nbar0.00022",
        #tag_mock="_nbar0.00054",
        idxs_obs=None,
    ),
}

# In-distribution test presets only. Values are (tag_biasparams_test, tag_noise_test) where
# tag_noise_test None means no noise fields; "__unit__" means "_noise_unit" + tag_params_test.
# For shame + noisy, bias stays ``_biasshame_*`` (fixed bias draw); only the noise-field tag is derived.
_TEST_SCENARIO_TAGS = {
    "coverage": {
        "noiseless": ("_biascoverage_p4_n1000", None),
        "noisy": ("_biasnoisecoverage_p9_n1000", "__unit__"),
        "noisym2": ("_biasnoisem2coverage_p7_n1000", "__unit__"),
    },
    "fixed_cosmo_shame_mean": {
        "noiseless": ("_biasshame_p0_n1", None),
        "noisy": ("_biasshame_p0_n1", "__unit__"),
        "noisym2": ("_bias_shame_noisem2_p3_n1000", "__unit__"),
    },
    "fixed_cosmo_shame_sample": {
        "noiseless": ("_biasshame_p0_n1", None),
        "noisy": ("_biasshame_p0_n1", "__unit__"),
        "noisym2": ("_bias_shame_noisem2_p3_n1000", "__unit__"),
    },
}


def resolve_test_scenario_tags(scenario, noise_mode, tag_params_test):
    """Return ``tag_biasparams_test`` and ``tag_noise_test`` for an in-distribution test preset."""
    if scenario not in _TEST_SCENARIO_TAGS:
        return {}
    if noise_mode not in NOISE_MODES:
        raise KeyError(f"Unknown noise_mode {noise_mode!r}; expected one of {NOISE_MODES}")
    bias_t, noise_t = _TEST_SCENARIO_TAGS[scenario][noise_mode]
    if noise_t == "__unit__":
        if tag_params_test is None:
            raise ValueError(f"scenario {scenario!r} needs tag_params_test for noisy unit-noise tag")
        if not tag_params_test.startswith("_"):
            raise ValueError(f"tag_params_test must start with '_', got {tag_params_test!r}")
        noise_t = "_noise_unit" + tag_params_test
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
                          sweep_name_override: str | None = None):
    """
    Generates a YAML configuration file for training.

    For ``run_mode`` ``sweep``, ``wandb_sweep_id`` (initially ``None``) and
    ``sweep_num_runs`` are written so re-running this same config can resume
    without environment variables.

    For ``run_mode`` ``best``, ``tag_sweep`` must match the sweep you optimized;
    ``sweep_name`` uses BX_SWEEP/N_TRAIN_SWEEP and ``tags_mask_for_sweep(statistics)``
    (``TAG_MASK_BISPEC_SWEEP`` on bispec), not the training ``tags_mask``.

    ``sweep_name_override``: if set, use this as ``sweep_name`` instead of the
    default (e.g. point at an existing noisy sweep while training on noisym2).
    """
    # bx is bias parameters per cosmo (1x, 2x, 4x, 8x, 16x, 32x)
    tags_mask = [""] * len(statistics) if tags_mask is None else list(tags_mask)
    assert len(tags_mask) == len(statistics), (
        f"tags_mask length ({len(tags_mask)}) must match statistics length ({len(statistics)}): "
        f"tags_mask={tags_mask!r}, statistics={statistics!r}"
    )

    tag_data_train = build_tag_data(
        data_mode, statistics, tags_mask, tag_params, tag_biasparams, tag_noise
    )
    tag_data_sweep = build_tag_data(
        data_mode,
        statistics,
        tags_mask_for_sweep(statistics),
        tag_params,
        tag_biasparams,
        tag_noise,
    )

    tag_inf_num = f'_bx{bx}_ntrain{n_train}'
    base_inf = tag_data_train + ('_rp' if reparameterize else '') + tag_inf_num
    tag_inf_num_sweep = f'_bx{BX_SWEEP}_ntrain{N_TRAIN_SWEEP}'
    base_inf_sweep = tag_data_sweep + ('_rp' if reparameterize else '') + tag_inf_num_sweep

    if run_mode == 'sweep':
        tag_inf = base_inf_sweep + f'_sweep{tag_sweep}'
        sweep_name = base_inf_sweep + f'_sweep{tag_sweep}'
    elif run_mode == 'best':
        # Output dir tag includes this bx/n_train; sweep_name is the completed sweep.
        tag_inf = base_inf + f'_best{tag_sweep}'
        sweep_name = base_inf_sweep + f'_sweep{tag_sweep}'
        if sweep_name_override is not None:
            sweep_name = sweep_name_override
    else:  # single
        tag_inf = base_inf
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
    if run_mode == "sweep":
        config["wandb_sweep_id"] = None
        config["sweep_num_runs"] = SWEEP_NUM_RUNS
            
    os.makedirs(dir_config, exist_ok=True)
    fn_config = f"{dir_config}/config{tag_inf}.yaml"
    if not overwrite and os.path.exists(fn_config):
        print(f"Config file already exists: {fn_config}")
        print("Set overwrite=True to overwrite the existing file.")
        return
    else:
        if os.path.exists(fn_config):
            print("Config file already exists but overwrite=True, overwriting. Hope you meant to do that!")
        with open(fn_config, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        print(f"Training config file written: {fn_config}")


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
                         data_mode_test="muchisimocks",
                         idxs_obs=None,
                         evaluate_mean=True,
                         tag_params_test="_shame_p0_n1000",
                         tag_biasparams_test="_biasshame_p0_n1",
                         tag_noise_test=None,
                         sweep_name_override: str | None = None):
    """
    Generates a YAML configuration file for testing.

    ``tags_mask`` is a list (length ``len(statistics)``) of per-statistic suffix strings
    (e.g. ``""`` or ``"_kb0.25"`` for bispec); joined into ``tag_masks`` for
    ``tag_data_train`` / ``tag_data_test`` / ``tag_inf_train``, matching training configs.

    If ``tag_sweep`` is set (e.g. ``'-rand30'``), ``tag_inf_train`` ends with
    ``_best<tag_sweep>`` (same checkpoint tree as training with that sweep's best run);
    if ``tag_sweep`` is ``None``, the plain training tag without ``_best`` is used.
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

    # Train checkpoint tag -> results_sbi/sbi<tag_inf_train> (optional _best from tag_sweep).
    tag_inf_num = f'_bx{bx}_ntrain{n_train}'
    base_inf_train = tag_data_train + ('_rp' if reparameterize else '') + tag_inf_num
    tag_inf_num_sweep = f'_bx{BX_SWEEP}_ntrain{N_TRAIN_SWEEP}'
    tag_data_sweep = build_tag_data(
        data_mode,
        statistics,
        tags_mask_for_sweep(statistics),
        tag_params,
        tag_biasparams,
        tag_noise,
    )
    base_inf_train_sweep = tag_data_sweep + ('_rp' if reparameterize else '') + tag_inf_num_sweep
    if tag_sweep is not None:
        # Same directory layout as training that finished a sweep and kept the best run.
        tag_inf_train = base_inf_train + f'_best{tag_sweep}'
        sweep_name = base_inf_train_sweep + f'_sweep{tag_sweep}'
        if sweep_name_override is not None:
            sweep_name = sweep_name_override
    else:
        tag_inf_train = base_inf_train
        sweep_name = None

    if evaluate_mean:
        tag_mean = '_mean'
    else:
        tag_mean = ''
    tag_test = f"_TRAIN{tag_inf_train}_TEST{tag_data_test}{tag_mean}"
    
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
        return
    else:
        if os.path.exists(fn_config):
            print("Config file already exists but overwrite=True, overwriting. Hope you meant to do that!")
        with open(fn_config, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        print(f"Testing config file written: {fn_config}")
        
        
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
                             idxs_obs=None,
                             evaluate_mean=False,
                             data_mode_test="shame",
                             tag_mock="_nbar0.00022",
                             sweep_name_override: str | None = None):
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
    tag_data_train_sweep = build_tag_data(
        data_mode,
        statistics,
        tags_mask_for_sweep(statistics),
        tag_params,
        tag_biasparams,
        tag_noise,
    )

    # Train checkpoint tag -> results_sbi/sbi<tag_inf_train> (optional _best from tag_sweep).
    tag_inf_num = f'_bx{bx}_ntrain{n_train}'
    base_inf_train = tag_data_train + ('_rp' if reparameterize else '') + tag_inf_num
    tag_inf_num_sweep = f'_bx{BX_SWEEP}_ntrain{N_TRAIN_SWEEP}'
    base_inf_train_sweep = tag_data_train_sweep + ('_rp' if reparameterize else '') + tag_inf_num_sweep
    if tag_sweep is not None:
        tag_inf_train = base_inf_train + f'_best{tag_sweep}'
        sweep_name = base_inf_train_sweep + f'_sweep{tag_sweep}'
        if sweep_name_override is not None:
            sweep_name = sweep_name_override
    else:
        tag_inf_train = base_inf_train
        sweep_name = None

    ### test tags
    tag_data_test = '_'+data_mode_test + tag_stats + tag_masks_train + tag_mock
    
    if evaluate_mean:
        tag_mean = '_mean'
    else:
        tag_mean = ''
    tag_test = f"_TRAIN{tag_inf_train}_TEST{tag_data_test}{tag_mean}"
    
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
        return
    else:
        if os.path.exists(fn_config):
            print("Config file already exists but overwrite=True, overwriting. Hope you meant to do that!")
        with open(fn_config, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        print(f"Testing config file written: {fn_config}")


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
    sweep_name_override: str | None = None,
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

    Pass ``tag_sweep`` (e.g. ``'-rand30'``) to point at the ``_best<tag_sweep>`` checkpoint;
    omit it to load the non-sweep training run.
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
    common = dict(
        dir_config=dir_config,
        overwrite=overwrite,
        statistics=statistics,
        n_train=n_train,
        bx=bx,
        tags_mask=tags_mask,
        tag_sweep=tag_sweep,
        sweep_name_override=sweep_name_override,
        **train_fields,
        **test_fields,
        **scenario_tags,
        **kwargs,
    )
    if ood:
        return generate_test_config_ood(**common)
    return generate_test_config(**common)


def generate_runlike_config(dir_config=str(DEFAULT_CONFIGS_RUNLIKE_DIR), overwrite=False):
    """
    Generates a YAML configuration file for likelihood-based inference.
    """
    #data_mode = 'emu'  # or 'muchisimocksPk'
    data_mode = 'muchisimocks'
    statistics = ['pk'] 
    # i think i should make these "test" because no training, just evaluation!
    tag_params = '_quijote_p0_n1000'
    tag_biasparams = '_b1000_p0_n1'

    # Parameters to vary
    n_cosmo_params_vary = 5  # Number of cosmological parameters to vary
    n_bias_params_vary = 0  # Number of bias parameters to vary
    cosmo_param_names_vary = utils_model.cosmo_param_names_ordered[:n_cosmo_params_vary]
    bias_param_names_vary = utils_model.biasparam_names_ordered[:n_bias_params_vary]
    mcmc_framework = 'dynesty'  # or 'emcee'
    evaluate_mean = True
    #idxs_obs = [0]  # or None for all or evaluate_mean=True
    idxs_obs = None
    
    tag_stats = f'_{"_".join(statistics)}'    
    tag_data = '_'+data_mode + tag_stats + tag_params + tag_biasparams
    if evaluate_mean:
        tag_mean = '_mean'
    else:
        tag_mean = ''

    tag_inf = f'{tag_data}{tag_mean}_pvary{len(cosmo_param_names_vary)}_bvary{len(bias_param_names_vary)}'

    config = {
        'data_mode': data_mode,
        'statistics': statistics,
        'tag_params': tag_params,
        'tag_biasparams': tag_biasparams,
        'tag_data': tag_data,
        'tag_inf': tag_inf,
        'cosmo_param_names_vary': cosmo_param_names_vary,
        'bias_param_names_vary': bias_param_names_vary,
        'mcmc_framework': mcmc_framework,
        'evaluate_mean': evaluate_mean,
        'idxs_obs': idxs_obs,
    }

    os.makedirs(dir_config, exist_ok=True)
    fn_config = f"{dir_config}/config{tag_inf}.yaml"
    if not overwrite and os.path.exists(fn_config):
        print(f"Config file already exists: {fn_config}")
        print("Set overwrite=True to overwrite the existing file.")
        return
    else:
        if os.path.exists(fn_config):
            print("Config file already exists but overwrite=True, overwriting. Hope you meant to do that!")
        with open(fn_config, "w") as file:
            yaml.dump(config, file, default_flow_style=False)
        print(f"Runlike config file written: {fn_config}")


def main():
    overwrite = False
    # Cosmo LH tag (see ``generate_params`` / data dirs); pair with noise_mode for bias+noise tags.
    tag_params_train = "_p5_n10000"
    #noise_mode = "noiseless"  # or "noisy" or "noisym2"; see NOISE_MODE_TRAIN_BIAS
    #noise_mode = "noisy"
    noise_mode = "noisym2"
    train_kw = resolve_train_tag_bundle(tag_params_train, noise_mode)

    # Optional: full sweep_name tag (under results_sbi/sbi<name>/). None = derive from train tags.
    # Fiducial noisy rand30 pk sweep while training on noisym2:
    # NOTE: we are not changing the saving tags! if we eventually 
    # do sweep for this noise model, we should manually change this saved training run
    sweep_name_override = (
        # "_muchisimocks_pk_p5_n10000_biasnoisenest_p9_n320000_noise_unit_p5_n10000" \
        # "_rp_bx32_ntrain10000_sweep-rand30"
        "_muchisimocks_pk_pgm_p5_n10000_biasnoisem2nest_p7_n320000_noise_unit_p5_n10000" \
        "_rp_bx32_ntrain10000_sweep-rand30"
    )
    # sweep_name_override = None

    # Training: run_mode 'single' | 'sweep' | 'best'; tag_sweep required for sweep/best (e.g. '-rand10').
    # Note: run_mode is only for training; testing is always 'load', and if tag_sweep is passed will use best
    
    #mode = "test"
    mode = "train"
    #run_mode = "single"
    #tag_sweep = None
    #run_mode = "sweep"
    run_mode = "best"
    tag_sweep = "-rand30"

    stat_arr = [
        #["pk"],
        ["pk", "pgm"],
        #["pk", "bispec"],
        #["pk", "bispec", "pgm"],
    ]
    # stat_arr = [
    #     ["pk", "bispec"],
    #     ["pk", "bispec", "pgm"],
    # ]
    tags_mask_arr = [["", "_kpgm0.25"]]
    #tag_mask_bispec_arr = ["_kb0.1", "_kb0.15", "_kb0.2", "_kb0.25", "_kb0.3", "_kb0.35", ""]
    #tags_mask_arr = [["", tag_mask_bispec] for tag_mask_bispec in tag_mask_bispec_arr]
    #tag_mask_pgm_arr = ["_kpgm0.1", "_kpgm0.15", "_kpgm0.2", "_kpgm0.25", "_kpgm0.3", "_kpgm0.35", ""]
    #tags_mask_arr = [["", tag_mask_pgm] for tag_mask_pgm in tag_mask_pgm_arr]
    #tag_mask_bispec_arr = ["_kb0.2", "_kb0.25", "_kb0.3", "_kb0.35", ""]
    #tag_mask_pgm_arr = ["_kpgm0.2", "_kpgm0.25", "_kpgm0.3", "_kpgm0.35", ""]
    #tags_mask_arr = [["", tag_mask_bispec, tag_mask_pgm] for tag_mask_bispec in tag_mask_bispec_arr for tag_mask_pgm in tag_mask_pgm_arr]
    #tags_mask_arr = [
        #["", "_kb0.1"],    
        #["", "_kpgm0.35"],    
        #["", ""],    
        #["", "_kb0.25", "_kpgm0.3"]
    #]
    

    n_train_arr = [10000]
    bx_arr = [32]
    #n_train_arr = [500, 1000, 2000, 4000, 6000, 8000, 10000]
    #bx_arr = [1, 2, 4, 8, 16, 32]

    # assert len(tags_mask_arr) == len(stat_arr), (
    #     "tags_mask_arr must have one entry per stat_arr row (aligned lengths and statistics)"
    # )
    for i,statistics in enumerate(stat_arr):
        for n_train in n_train_arr:
            for bx in bx_arr:
                for tags_mask in tags_mask_arr:
                    #tags_mask = [tag_mask_pgm if s == "pgm" else "" for s in statistics]
                    if mode == "train":
                        generate_train_config(
                            overwrite=overwrite,
                            statistics=statistics,
                            tags_mask=tags_mask,
                            n_train=n_train,
                            bx=bx,
                            run_mode=run_mode,
                            tag_sweep=tag_sweep,
                            sweep_name_override=sweep_name_override,
                            **train_kw,
                        )
                    elif mode == "test":
                        #for test_name in PARAM_SETS_TEST:
                        for test_name in ["ood"]:
                        #for test_name in ["coverage", "fixed_cosmo_shame_mean"]:
                            generate_test_config_from_preset(
                                test_name,
                                tag_params=tag_params_train,
                                noise_mode=noise_mode,
                                overwrite=overwrite,
                                statistics=statistics,
                                n_train=n_train,
                                bx=bx,
                                tags_mask=tags_mask,
                                tag_sweep=tag_sweep,
                                sweep_name_override=sweep_name_override,
                            )
    # generate_runlike_config(overwrite=overwrite)


if __name__ == "__main__":
    main()
