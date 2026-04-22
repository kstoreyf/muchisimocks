import os
import re
import shutil
from types import SimpleNamespace

import numpy as np
import pathlib
import pickle
import time
import tempfile
import yaml
import wandb

from sbi.inference import NPE
from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn
import torch

import paths
import scaler_custom as scl
import generate_params as genp
import utils_inference


# Must match sweep_config['parameters'] and agent wandb.config.update in run().
_WANDB_SWEEP_PARAMETER_KEYS = (
    "learning_rate",
    "hidden_features",
    "training_batch_size",
    "num_transforms",
)

# Subdirs under ``dir_sbi`` for sweep trials use ``wandb.run.id`` (lowercase alnum).
_WANDB_RUN_ID_DIR_RE = re.compile(r"^[a-z0-9]{8,}$")


def wandb_sweep_best_run(sweep, *, minimize: str):
    """
    ``wandb.apis.public.Sweep.best_run`` only accepts ``order=``, not ``minimize=``.
    Minimizing a summary metric is ``order='+<name>'`` (see ``QueryGenerator``).

    Raises ``ValueError`` if the selected run has no ``minimize`` value in summary
    (metric never logged); do not treat an unlogged metric as a comparable best.
    """
    run = sweep.best_run(order=f"+{minimize}")
    if run is None:
        raise ValueError(
            "No run returned for sweep id=%r (empty sweep or filter)."
            % (getattr(sweep, "id", None),)
        )
    sm = dict(run.summary)
    val = sm.get(minimize)
    if val is None:
        raise ValueError(
            "Best run %s has no summary[%r] logged; cannot select best without that metric."
            % (run.id, minimize)
        )
    return run


def _read_best_run_id_from_dir_sbi(dir_sbi):
    """
    W&B run id from ``best_run.txt`` in the given results directory (written by
    ``choose_best_run.py`` after ranking + stall tests).
    """
    fn = os.path.join(dir_sbi, "best_run.txt")
    if not os.path.isfile(fn):
        raise FileNotFoundError(
            "Missing best run file %s. Run choose_best_run.py for this sweep first "
            "(it writes best_run.txt with the chosen W&B run id)."
            % fn
        )
    with open(fn, encoding="utf-8") as f:
        line = f.readline()
    parts = line.split()
    run_id = parts[0] if parts else ""
    if not run_id:
        raise ValueError(
            "Empty or invalid W&B run id in %s (expected one line: run id only)."
            % fn
        )
    if _WANDB_RUN_ID_DIR_RE.match(run_id) is None:
        raise ValueError(
            "Unexpected run id %r in %s (expected a wandb run id)." % (run_id, fn)
        )
    return run_id


def _wandb_sweep_api_path(sweep_id_or_path, project_name):
    """
    W&B API expects ``entity/project/sweep_id``. Accept that full string, or a
    short sweep id with default entity from the logged-in user.
    """
    s = (sweep_id_or_path or "").strip()
    if s.count("/") >= 2:
        return s
    api = wandb.Api()
    entity = api.default_entity
    return f"{entity}/{project_name}/{s}"


def _parse_wandb_sweep_api_path(sweep_api_path):
    """Split ``entity/project/sweep_id`` into a triple."""
    parts = (sweep_api_path or "").strip().split("/")
    if len(parts) != 3:
        raise ValueError(
            "wandb_sweep_id must be sweep_id or entity/project/sweep_id; got %r"
            % sweep_api_path
        )
    return parts[0], parts[1], parts[2]


def _write_wandb_sweep_id_to_yaml(yaml_path, sweep_api_path):
    """Persist ``wandb_sweep_id`` so the next job can resume without env vars."""
    p = pathlib.Path(yaml_path)
    if not p.is_file():
        print("wandb: training config not found (%s); skip writing sweep id" % p)
        return
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        print("wandb: training config is not a mapping; skip writing sweep id")
        return
    if data.get("wandb_sweep_id") == sweep_api_path:
        return
    data["wandb_sweep_id"] = sweep_api_path
    fd, tmp_name = tempfile.mkstemp(
        suffix=".yaml", dir=str(p.parent), text=True
    )
    try:
        with os.fdopen(fd, "w") as tmp:
            yaml.dump(data, tmp, default_flow_style=False, sort_keys=False)
        pathlib.Path(tmp_name).replace(p)
    except OSError as e:
        print("wandb: could not write sweep id to config: %s" % e)
        try:
            pathlib.Path(tmp_name).unlink(missing_ok=True)
        except OSError:
            pass
    else:
        print("wandb: saved wandb_sweep_id to %s" % p)


def _delete_crashed_runs_in_sweep(sweep_api_path, dir_sbi=None):
    """
    Delete non-running failed runs in an existing sweep before resuming.

    If ``dir_sbi`` is set, also remove ``dir_sbi/<run_id>/`` for each deleted run.
    """
    api = wandb.Api()
    sweep = api.sweep(sweep_api_path)
    to_remove = [
        r
        for r in sweep.runs
        if str(getattr(r, "state", "")).lower() in {"crashed", "failed", "killed"}
    ]
    n_deleted = 0
    n_local = 0
    root = pathlib.Path(dir_sbi) if dir_sbi else None
    for run in to_remove:
        rid = run.id
        run.delete()
        n_deleted += 1
        if root is not None and _WANDB_RUN_ID_DIR_RE.match(str(rid)):
            p = root / rid
            if p.is_dir():
                shutil.rmtree(p)
                n_local += 1
    print(
        "wandb: deleted %s crashed/failed/killed runs from %s (local run dirs removed: %s)"
        % (n_deleted, sweep_api_path, n_local if root is not None else "n/a")
    )


def _delete_orphaned_local_sweep_run_dirs(dir_sbi, sweep_api_path):
    """
    Remove ``dir_sbi/<run_id>/`` subdirs whose ``run_id`` is not a run in this sweep
    on W&B (e.g. runs removed manually in the UI).
    """
    api = wandb.Api()
    sweep = api.sweep(sweep_api_path)
    active_ids = {str(r.id) for r in sweep.runs}
    root = pathlib.Path(dir_sbi)
    if not root.is_dir():
        return
    n_removed = 0
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if not _WANDB_RUN_ID_DIR_RE.match(name):
            continue
        if name in active_ids:
            continue
        shutil.rmtree(child)
        n_removed += 1
        print("wandb: removed orphaned local sweep run dir %s" % child)
    if n_removed:
        print(
            "wandb: removed %s orphaned local sweep run dir(s) under %s"
            % (n_removed, dir_sbi)
        )


def _wandb_count_finished_success_runs(sweep_api_path):
    """
    Count runs in a successful terminal state. Used with ``sweep_num_runs`` so
    ``wandb.agent(..., count=...)`` can stop in-process (``count=None`` does not).
    """
    api = wandb.Api()
    sweep = api.sweep(sweep_api_path)
    n = 0
    for run in sweep.runs:
        state = str(getattr(run, "state", "")).lower()
        if state in {"finished", "completed"}:
            n += 1
    return n


def _fit_config_from_training_dict(
    raw,
    max_epochs_default,
    validation_fraction_default,
    *,
    source,
):
    """
    Shared by W&B run config and ``config.pkl`` from a sweep trial (``fit_model``
    saves a plain dict via ``_training_config_to_plain_dict``). Requires the four sweep
    sample keys; optional ``max_epochs``, ``model_type``, ``validation_fraction``.
    """
    raw = {k: v for k, v in dict(raw).items() if not str(k).startswith("_")}
    missing = [k for k in _WANDB_SWEEP_PARAMETER_KEYS if raw.get(k) is None]
    if missing:
        raise ValueError(
            "%s: missing keys %s (need all of %s). Saw: %s"
            % (source, missing, _WANDB_SWEEP_PARAMETER_KEYS, sorted(raw.keys()))
        )
    return SimpleNamespace(
        learning_rate=float(raw["learning_rate"]),
        hidden_features=int(raw["hidden_features"]),
        training_batch_size=int(raw["training_batch_size"]),
        num_transforms=int(raw["num_transforms"]),
        model_type=raw.get("model_type") or "maf",
        max_epochs=int(raw.get("max_epochs", max_epochs_default)),
        validation_fraction=float(
            raw.get("validation_fraction", validation_fraction_default)
        ),
    )


def _fit_config_from_wandb_run(run, max_epochs_default, validation_fraction_default):
    """See ``_fit_config_from_training_dict``."""
    raw = {k: v for k, v in dict(run.config).items() if not str(k).startswith("_")}
    return _fit_config_from_training_dict(
        raw,
        max_epochs_default,
        validation_fraction_default,
        source="W&B run %s" % run.id,
    )


def _fit_config_from_config_pkl(path, max_epochs_default, validation_fraction_default):
    """Load sweep hparams from a trial's ``config.pkl`` (same schema as W&B config)."""
    with open(path, "rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        raise ValueError("%s: expected a dict, got %s" % (path, type(raw).__name__))
    return _fit_config_from_training_dict(
        raw,
        max_epochs_default,
        validation_fraction_default,
        source=path,
    )


def _training_config_to_plain_dict(config):
    """
    Snapshot hyperparameters for ``config.pkl``.

    ``wandb.config`` stores user keys internally; ``vars(wandb.config)`` only has
    private attributes like ``_items``, so filtering ``k.startswith('_')`` yields
    ``{}``. Use the mapping API (``dict(config)``) when possible; fall back to
    ``vars`` for :class:`types.SimpleNamespace` (``dict(ns)`` raises).
    """
    if isinstance(config, dict):
        raw = config
    else:
        try:
            raw = dict(config)
        except (TypeError, ValueError):
            raw = vars(config)
    return {k: v for k, v in raw.items() if not str(k).startswith("_")}


class SBIModel():

    def __init__(self, theta_train=None, y_train_unscaled=None,
                     theta_test=None, y_test_unscaled=None,
                     statistics=None, param_names=None, dict_bounds=None, 
                     run_mode='single', tag_sbi='', n_threads=1, 
                     sweep_name=None, overwrite=False,
                     # True iff bx/n_train match BX_SWEEP/N_TRAIN_SWEEP and tags_mask matches
                     # tags_mask_for_sweep(statistics); run_mode best: copy sweep artifact vs retrain.
                     matches_sweep_model=False,
                     wandb_sweep_id=None,
                     sweep_num_runs=None,
                     wandb_config_yaml_path=None,
                     ):

        self.dir_sbi = str(paths.DIR_RESULTS / "results_sbi" / f"sbi{tag_sbi}")
        p = pathlib.Path(self.dir_sbi)
        p.mkdir(parents=True, exist_ok=True)
        
        self.theta_train = theta_train
        self.y_train_unscaled = y_train_unscaled

        self.theta_test = theta_test
        self.y_test_unscaled = y_test_unscaled
        
        # param_names are required (saved model doesn't store them)
        assert param_names is not None, 'need parameter names'
        self.param_names = param_names
        assert len(statistics) > 0, 'Pass statistics! (Needed for scaler)'
        self.statistics = statistics
        
        if self.y_train_unscaled is not None:
            self.setup_scalers_y()
            self.n_dim = self.y_train.shape[1]
            print('ndim:', self.n_dim)
        elif self.y_test_unscaled is not None:
            self.n_dim = np.sum([y_test_i.shape[1] for y_test_i in self.y_test_unscaled])

        if self.y_train_unscaled is None:
            self.load_scalers_y()
            
        if self.theta_train is not None:
            self.n_params = self.theta_train.shape[1]
        elif self.theta_test is not None:
            self.n_params = self.theta_test.shape[1]
        
        self.run_mode = run_mode
        if self.run_mode != 'load':
            assert dict_bounds is not None, 'need dict_bounds if training model'
        self.dict_bounds = dict_bounds
        self.sweep_name = sweep_name
        
        self.n_threads = n_threads
        self.overwrite = overwrite
        self.matches_sweep_model = matches_sweep_model
        self.wandb_sweep_id = wandb_sweep_id
        self.sweep_num_runs = int(sweep_num_runs) if sweep_num_runs is not None else None
        self.wandb_config_yaml_path = (
            str(pathlib.Path(wandb_config_yaml_path).resolve())
            if wandb_config_yaml_path
            else None
        )

    def _sweep_results_dir_for_best_run_txt(self):
        """
        Sweep results directory ``results_sbi/sbi{sweep_name}`` where
        ``choose_best_run.py`` writes ``best_run.txt``. Same string as W&B ``sweep_name``
        / training sweep ``tag_inf``. Not ``self.dir_sbi`` (the ``_best...`` output tree).
        """
        if not self.sweep_name:
            raise ValueError(
                "run_mode=best requires sweep_name (sweep folder = sbi<sweep_name>)."
            )
        return str(
            paths.DIR_RESULTS / "results_sbi" / ("sbi" + self.sweep_name)
        )

        
    def run(self, max_epochs=2000):
        validation_fraction = 0.1

        print("run mode:", self.run_mode)
        print("sweep name:", self.sweep_name)
        print("dir_sbi:", self.dir_sbi)
        
        project_name = 'muchisimocks-sbi'
        
        if self.run_mode == 'sweep':
            if self.sweep_num_runs is None:
                raise ValueError("run_mode='sweep' requires sweep_num_runs in the config")
            wandb.login()
            resume_id = (self.wandb_sweep_id or "").strip()
            sweep_config = {
                'name': self.sweep_name,
                'method': 'random',
                'run_cap': self.sweep_num_runs,
                'metric': {
                    # Logged in fit_model as min(epoch validation loss); matches sweep ranking.
                    'name': 'best_validation_loss',
                    'goal': 'minimize'
                },
                'parameters': {
                    'learning_rate': {'values': [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]},
                    'hidden_features': {'values': [32, 64, 128, 256]},
                    'training_batch_size': {'values': [32, 64, 128, 256]},
                    'num_transforms': {'values': [4, 6, 8, 10]},
                }
            }
            if resume_id:
                sweep_api_path = _wandb_sweep_api_path(resume_id, project_name)
                agent_entity, agent_project, sweep_id = _parse_wandb_sweep_api_path(
                    sweep_api_path
                )
                _delete_crashed_runs_in_sweep(sweep_api_path, self.dir_sbi)
                print(
                    "wandb sweep resume: path=%s target_total_finished_runs=%s"
                    % (sweep_api_path, self.sweep_num_runs),
                    flush=True,
                )
            else:
                sweep_id = wandb.sweep(sweep_config, project=project_name)
                api = wandb.Api()
                agent_entity = api.default_entity
                agent_project = project_name
                sweep_api_path = f"{agent_entity}/{agent_project}/{sweep_id}"
                if self.wandb_config_yaml_path:
                    _write_wandb_sweep_id_to_yaml(
                        self.wandb_config_yaml_path, sweep_api_path
                    )
                print(
                    (
                        "Started wandb sweep %s. Re-run the same training command to resume "
                        "(wandb_sweep_id is in the config if it was saved)."
                    )
                    % sweep_api_path,
                    flush=True,
                )

            _delete_orphaned_local_sweep_run_dirs(self.dir_sbi, sweep_api_path)

            n_finished = _wandb_count_finished_success_runs(sweep_api_path)
            count = max(0, int(self.sweep_num_runs) - n_finished)
            print(
                "wandb: agent count=%s (target finished runs=%s, already finished=%s)"
                % (count, self.sweep_num_runs, n_finished),
                flush=True,
            )
            if count == 0:
                print(
                    "wandb: skipping wandb.agent (finished runs=%s, target=%s)."
                    % (n_finished, self.sweep_num_runs),
                    flush=True,
                )
                wandb.finish()
                return

            def _fit_model_wandb():
                wandb.init(project=agent_project, entity=agent_entity)
                wandb.config.update({
                    'max_epochs': max_epochs,
                    'model_type': 'maf',
                    'validation_fraction': validation_fraction,
                })
                run_dir = os.path.join(self.dir_sbi, wandb.run.id)
                self.fit_model(wandb.config, save_model=True, save_dir=run_dir)
                artifact = wandb.Artifact(name=f"sbi-posterior-{wandb.run.id}", type="model")
                artifact.add_dir(local_path=run_dir)
                wandb.run.log_artifact(artifact)

            wandb.agent(
                sweep_id,
                function=_fit_model_wandb,
                count=count,
                entity=agent_entity,
                project=agent_project,
            )
            wandb.finish()
            
        elif self.run_mode == 'best':
            # matches_sweep_model True: copy trained run into dir_sbi (local sweep subdir if
            # present, else W&B artifact download). False: retrain from W&B run hparams.
            # Best run id: read from best_run.txt under results_sbi/sbi{sweep_name} (not
            # self.dir_sbi, which is the _best... output tree).
            try:
                sweep_dir = self._sweep_results_dir_for_best_run_txt()
            except ValueError as e:
                raise SystemExit(str(e)) from e
            print(
                "run_mode=best: looking for best_run.txt in sweep dir %s "
                "(output dir_sbi=%s, sweep_name=%r)"
                % (sweep_dir, self.dir_sbi, self.sweep_name),
                flush=True,
            )
            try:
                best_run_id = _read_best_run_id_from_dir_sbi(sweep_dir)
            except (FileNotFoundError, ValueError) as e:
                raise SystemExit(str(e)) from e
            best_run_txt = os.path.join(sweep_dir, "best_run.txt")
            print(
                "run_mode=best: read chosen W&B run id %r from %s"
                % (best_run_id, best_run_txt),
                flush=True,
            )

            local_run_dir = os.path.join(sweep_dir, best_run_id)
            local_posterior = os.path.join(local_run_dir, "posterior.p")
            local_cfg_pkl = os.path.join(local_run_dir, "config.pkl")

            if self.matches_sweep_model and os.path.isfile(local_posterior):
                print(
                    "run_mode=best: hyperparameters/source = sweep trial checkpoint (copy local posterior; "
                    "same bx/n_train and tags_mask as sweep).",
                    flush=True,
                )
                print(
                    "run_mode=best: found trained model under sweep dir — copying to %s "
                    "(skipping wandb login / artifact download)."
                    % self.dir_sbi,
                    flush=True,
                )
                os.makedirs(self.dir_sbi, exist_ok=True)
                for fn in ["posterior.p", "inference.p", "param_names.txt", "config.pkl"]:
                    src = os.path.join(local_run_dir, fn)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(self.dir_sbi, fn))
                for f in pathlib.Path(local_run_dir).glob("scaler_y_*.p"):
                    shutil.copy2(f, os.path.join(self.dir_sbi, f.name))
                print("Copied best sweep model to %s" % self.dir_sbi)
            elif self.matches_sweep_model:
                wandb.login()
                api = wandb.Api()
                entity = api.default_entity
                run_path = "%s/%s/%s" % (entity, project_name, best_run_id)
                best_run = api.run(run_path)
                print(
                    "run_mode=best: resolved sweep run %s (W&B path %s)"
                    % (best_run.id, run_path),
                    flush=True,
                )
                model_artifacts = [
                    a for a in best_run.logged_artifacts() if a.type == "model"
                ]
                assert model_artifacts, (
                    "No model artifact for best run %s (no local posterior at %s)"
                    % (best_run.id, local_posterior)
                )
                artifact = model_artifacts[0]
                download_root = artifact.download()
                os.makedirs(self.dir_sbi, exist_ok=True)
                for fn in ["posterior.p", "inference.p", "param_names.txt", "config.pkl"]:
                    src = os.path.join(download_root, fn)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(self.dir_sbi, fn))
                for f in pathlib.Path(download_root).glob("scaler_y_*.p"):
                    shutil.copy2(f, os.path.join(self.dir_sbi, f.name))
                print("Copied best sweep model to %s" % self.dir_sbi)
            else:
                print(
                    "run_mode=best: retrain path — apply best sweep trial hyperparameters to this job's "
                    "training data (output dir_sbi=%s; trial %s under %s)."
                    % (self.dir_sbi, best_run_id, sweep_dir),
                    flush=True,
                )
                cfg = None
                if os.path.isfile(local_cfg_pkl):
                    try:
                        cfg = _fit_config_from_config_pkl(
                            local_cfg_pkl, max_epochs, validation_fraction
                        )
                    except ValueError as e:
                        raise SystemExit(str(e)) from e
                    print(
                        "run_mode=best: hparams from %s (skip W&B run fetch)"
                        % local_cfg_pkl,
                        flush=True,
                    )
                if cfg is None:
                    wandb.login()
                    api = wandb.Api()
                    entity = api.default_entity
                    run_path = "%s/%s/%s" % (entity, project_name, best_run_id)
                    best_run = api.run(run_path)
                    print(
                        "run_mode=best: resolved sweep run %s for hparams (%s)"
                        % (best_run.id, run_path),
                        flush=True,
                    )
                    cfg = _fit_config_from_wandb_run(
                        best_run, max_epochs, validation_fraction
                    )
                wandb.login()
                wandb.init(
                    project=project_name,
                    config=vars(cfg),
                    name="retrain-besthp-%s" % best_run_id,
                )
                self.fit_model(cfg, save_model=True)
                wandb.finish()
            
        elif self.run_mode == 'single':
            print(
                "run_mode=single: hyperparameters from built-in defaults in sbi_model.SBIModel.run "
                "(learning_rate, hidden_features, …); see fit_model wandb.config print below.",
                flush=True,
            )
            wandb.login()
            # Use default config for single run
            config = {
                'learning_rate': 1e-3,
                'hidden_features': 128,
                'training_batch_size': 64,
                'num_transforms': 5,
                'model_type': 'maf',
                'max_epochs': max_epochs,
                'validation_fraction': validation_fraction,
            }
            wandb.init(project=project_name, config=config)
            self.fit_model(wandb.config, save_model=True)
            wandb.finish()

        elif self.run_mode == 'load':
            self.load_posterior()
            self.load_param_names()
        else:
            raise ValueError(f"run_mode {self.run_mode} not recognized")
            
    def fit_model(self, config, save_model=True, save_dir=None):
        """
        Fit the SBI model using the specified configuration.

        If save_dir is provided, save to that directory instead of self.dir_sbi
        (used for sweep runs so each run gets its own subdirectory).
        """
        
        print(f"Fitting model for dir_sbi={self.dir_sbi}, run_mode={self.run_mode}, sweep_name={self.sweep_name}")
        print("wandb.config:", config)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Optimize PyTorch settings
        if device == "cpu":
            print(f"Using CPU with {self.n_threads} threads")
            torch.set_num_threads(self.n_threads)  # Use multiple CPU threads
        elif device == "cuda":
            print(f"Using GPU with {torch.cuda.device_count()} devices")
        
        # get prior
        l_bounds = np.array([self.dict_bounds[pn][0] for pn in self.param_names])
        u_bounds = np.array([self.dict_bounds[pn][1] for pn in self.param_names])
        prior = BoxUniform(low=torch.from_numpy(l_bounds),
                           high=torch.from_numpy(u_bounds))

        print(self.dict_bounds)
        print("Setting up inference")
        
        # Pull hyperparameters from config
        learning_rate = config.learning_rate
        training_batch_size = config.training_batch_size
        hidden_features = config.hidden_features
        model_type = config.model_type
        max_epochs = config.max_epochs
        num_transforms = getattr(config, 'num_transforms', 5) # sbi default is 5

        density_estimator_build_fun = posterior_nn(
            model=model_type,
            hidden_features=hidden_features,
            num_transforms=num_transforms,
        )

        validation_fraction = float(getattr(config, "validation_fraction", 0.1))
        print("theta_train shape:", self.theta_train.shape)
        print(f"validation_fraction (sbi internal split): {validation_fraction}")
        
        inference = NPE(prior=prior, density_estimator=density_estimator_build_fun)
        inference = inference.append_simulations(
            torch.tensor(self.theta_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.float32),
            )
        
        print(f"Training with {self.n_threads} threads")
        start = time.time()

        density_estimator = inference.train(
            max_num_epochs=max_epochs,
            training_batch_size=training_batch_size,
            validation_fraction=validation_fraction,
            learning_rate=learning_rate,
            show_train_summary=True,
        )
        # sbi reloads best-val weights inside the training loop only when early stopping
        # triggers; if we stop because max_num_epochs is hit first, weights can still be
        # from the last epoch. Always restore the best checkpoint (no-op if already loaded).
        best_sd = getattr(inference, "_best_model_state_dict", None)
        if best_sd is not None and inference._neural_net is not None:
            inference._neural_net.load_state_dict(best_sd)
        print("Trained!")
        end = time.time()
        print(f"Training time: {end - start:.2f}s = {(end - start) / 60:.2f} min (max_epochs={max_epochs}, n_threads={self.n_threads})")

        # Log final losses to wandb (best_validation_loss is the sweep objective key)
        train_log = inference._summary
        if train_log and len(train_log['training_loss']) > 0:
            final_training_loss = train_log['training_loss'][-1] if train_log['training_loss'] is not None else None
            final_validation_loss = train_log['validation_loss'][-1] if train_log['validation_loss'] is not None else None
            if final_training_loss is not None:
                wandb.log({"final_training_loss": final_training_loss})
            if final_validation_loss is not None:
                wandb.log({"final_validation_loss": final_validation_loss})
            val_series = train_log.get("validation_loss")
            if val_series:
                wandb.log({"best_validation_loss": float(min(val_series))})
        
        print("Building posterior")
        self.posterior = inference.build_posterior(density_estimator)
        print(self.posterior)

        out_dir = save_dir if save_dir is not None else self.dir_sbi

        # save model if requested (e.g., for best model from sweep or single run)
        if save_model:
            os.makedirs(out_dir, exist_ok=True)
            if save_dir is not None:
                # Copy scalers so the run dir is self-contained (for artifact)
                for stat in self.statistics:
                    src = f"{self.dir_sbi}/scaler_y_{stat}.p"
                    if os.path.exists(src):
                        shutil.copy2(src, f"{out_dir}/scaler_y_{stat}.p")
            with open(f"{out_dir}/posterior.p", "wb") as f:
                pickle.dump(self.posterior, f)
            with open(f"{out_dir}/inference.p", "wb") as f:
                pickle.dump(inference, f)
            with open(f"{out_dir}/param_names.txt", "w") as f:
                np.savetxt(f, self.param_names, fmt="%s")
            config_dict = _training_config_to_plain_dict(config)
            with open(f"{out_dir}/config.pkl", "wb") as f:
                pickle.dump(config_dict, f)
            print(f"Saved model to {out_dir}")

            
    def load_posterior(self):
        fn_posterior = f'{self.dir_sbi}/posterior.p'
        assert os.path.exists(fn_posterior), f"posterior.p not found in {self.dir_sbi}"
        print(f"Loading posterior from {fn_posterior}")
        with open(fn_posterior, "rb") as f:
            self.posterior = pickle.load(f)

    def load_param_names(self):
        fn_param_names = f'{self.dir_sbi}/param_names.txt'
        assert os.path.exists(fn_param_names), f"param_names.txt not found in {self.dir_sbi}"
        print(f"Loading param_names from {fn_param_names}")
        with open(fn_param_names, "r") as f:
            self.param_names = np.loadtxt(f, dtype=str)


    def setup_scalers_y(self):
        self.scalers_y = []
        self.y_train = np.empty((len(self.y_train_unscaled[0]), 0))
        if self.y_test_unscaled is not None:
            self.y_test = np.empty((len(self.y_test_unscaled[0]), 0))
        print(f"y_train shape: {self.y_train.shape}")
                
        for i, statistic in enumerate(self.statistics):
            
            func_scaler_y = utils_inference.statistics_scaler_funcs[statistic]
            
            scaler_y = scl.Scaler(func_scaler_y)
            print("statistic:", statistic)
            print(f"min and max before scaling: {np.min(self.y_train_unscaled[i]):3f}, {np.max(self.y_train_unscaled[i]):3f}")
            
            scaler_y.fit(self.y_train_unscaled[i])
            self.scalers_y.append(scaler_y)
            
            y_train_i = scaler_y.scale(self.y_train_unscaled[i])

            self.y_train = np.concatenate((self.y_train, y_train_i), axis=1)
            if self.y_test_unscaled is not None:
                y_test_i = scaler_y.scale(self.y_test_unscaled[i])
                self.y_test = np.concatenate((self.y_test, y_test_i), axis=1)
            
            print(f"min and max after scaling (func={func_scaler_y}): {np.min(scaler_y.scale(self.y_train_unscaled[i])):3f}, {np.max(scaler_y.scale(self.y_train_unscaled[i])):3f}")
            
            # save scaler - need pickle for custom object!!
            fn_scaler_y = f'{self.dir_sbi}/scaler_y_{statistic}.p'
            with open(fn_scaler_y, "wb") as f:
                pickle.dump(scaler_y, f)
                
        if self.y_test_unscaled is not None:
            print(f"y_test shape: {self.y_test.shape}")
                

    def load_scalers_y(self):

        self.scalers_y = []
        for i, statistic in enumerate(self.statistics):
            fn_scaler_y = f'{self.dir_sbi}/scaler_y_{statistic}.p'
            with open(fn_scaler_y, "rb") as f:
                self.scalers_y.append(pickle.load(f))
        print(f"Loaded scalers from {self.dir_sbi}")
        
    
    def evaluate(self, y_obs_unscaled, n_samples=10000):
        # convergence tests show 10,000 is probably good enough, tho for some
        # parameters there is fluctuation bw 10k, 30k, 100k
        # (see notebooks/2025-01-24_inference_muchisimocksPk.ipynb)
        if y_obs_unscaled[0].ndim == 1:
            n_data = 1
        else:
            n_data = y_obs_unscaled[0].shape[0]
        y_obs = np.empty((n_data, 0))
        for i, y_obs_unscaled_i in enumerate(y_obs_unscaled):
            if y_obs_unscaled[0].ndim == 1:
                y_obs_unscaled_i = np.expand_dims(y_obs_unscaled_i, axis=0)
            y_test_i = self.scalers_y[i].scale(y_obs_unscaled_i)
            y_obs = np.concatenate((y_obs, y_test_i), axis=1
                                   )
        print(f"Testing on y_obs with shape: {y_obs.shape}")
        start = time.time()
        # model is built with float32 so need the data to be here too
        y_obs = np.float32(np.array(y_obs))
        # using samples_batched bc always putting into 2d first (if were 2d, "samples")
        
        samples = self.posterior.sample_batched((n_samples,), x=y_obs)
        print(f"Time to sample (y_obs.shape={y_obs.shape}, n_samples={n_samples}): {time.time() - start:.2f}s = {(time.time() - start) / 60:.2f} min")
        return samples
    
    
    
    def evaluate_test_set(self, y_test_unscaled=None, tag_test_eval='',
                          n_samples=10000, checkpoint_every=100,
                          resume=True, n_test_eval=100):
        
        # y_test_unscaled is an array of length n_statistics, each with shape (n_test, n_dim);
        # concatenate inside evaluate bc we need to scale based on each stat
        print(f"Evaluating test set with tag {tag_test_eval}")
        if y_test_unscaled is None:
            y_test_unscaled = self.y_test_unscaled
        
        # Limit to subset of test samples if n_test_eval is specified
        if n_test_eval is not None and y_test_unscaled[0].ndim > 1:
            n_available = len(y_test_unscaled[0])
            n_to_use = min(n_test_eval, n_available)
            if n_to_use < n_available:
                print(f"Limiting test set from {n_available} to {n_to_use} samples")
                y_test_unscaled = [y_stat[:n_to_use] for y_stat in y_test_unscaled]
        
        # Set up file paths
        fn_samples_test_pred = f'{self.dir_sbi}/samples_test{tag_test_eval}_pred.npy'
        fn_samples_test_pred_inprogress = f'{self.dir_sbi}/samples_test{tag_test_eval}_pred_inprogress.npy'
        checkpoint_file = f"{self.dir_sbi}/checkpoint_samples_test{tag_test_eval}.txt"

        if y_test_unscaled[0].ndim == 1:
            samples_total = 1
        else:
            samples_total = len(y_test_unscaled[0])
        samples_completed = 0
        existing_samples = None
        
        print(f"Checkpoint file: {checkpoint_file}")
        
        if resume and not self.overwrite:
                        
            # Check if final file already exists (complete run)
            if os.path.exists(fn_samples_test_pred):
                existing_samples = np.load(fn_samples_test_pred)
                if existing_samples.shape[0] >= samples_total:
                    print(f"Found complete samples file: {fn_samples_test_pred} with {existing_samples.shape[0]} samples")
                    return
            
            # Check existing in-progress samples file
            if os.path.exists(fn_samples_test_pred_inprogress):
                existing_samples = np.load(fn_samples_test_pred_inprogress)
                samples_completed = existing_samples.shape[0]
                print(f"Found existing in-progress samples file with {samples_completed} samples")
                
            # Check checkpoint file for consistency
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    checkpoint_count = int(f.read().strip())
                print(f"Checkpoint file indicates {checkpoint_count} completed samples")
                
                # Use the checkpoint count if consistent, otherwise trust the samples file
                if existing_samples is not None and checkpoint_count == existing_samples.shape[0]:
                    samples_completed = checkpoint_count
                elif existing_samples is not None:
                    print(f"Checkpoint mismatch - using samples file count: {existing_samples.shape[0]}")
                    samples_completed = existing_samples.shape[0]
                else:
                    samples_completed = checkpoint_count
            
            if samples_completed >= samples_total:
                print(f"All {samples_total} samples already completed!")
                return
                
            if samples_completed > 0:
                print(f"Resuming from {samples_completed} completed samples")
        
        if self.overwrite:
            print("Overwrite is True - starting fresh")
            samples_completed = 0
            existing_samples = None
        
        start_time = time.time()
        
        # Sample in batches
        remaining_samples = samples_total - samples_completed
        
        try:
            while remaining_samples > 0:
                batch_size = min(checkpoint_every, remaining_samples)
                print(f"Sampling batch of {batch_size} samples ({samples_completed}/{samples_total} completed)")
                
                # Extract the chunk of observations we need to process
                start_idx = samples_completed
                end_idx = samples_completed + batch_size
                
                # Get the batch of y_test_unscaled data for this chunk
                if y_test_unscaled[0].ndim == 1:
                    # Single observation case - just use the same observation for all samples
                    y_test_unscaled_batch = y_test_unscaled
                else:
                    # Multiple observations case - extract the chunk from each statistic's array
                    y_test_unscaled_batch = [y_stat[start_idx:end_idx] for y_stat in y_test_unscaled]
                
                batch_start = time.time()
                # Use the existing evaluate method for this batch
                print(f"Evaluating batch {start_idx} to {end_idx}")
                batch_samples = self.evaluate(y_test_unscaled_batch, n_samples=n_samples)
                batch_end = time.time()
                
                print(f"Batch samples shape: {batch_samples.shape}")
                
                # Combine with existing samples if any (concatenate along axis=1 for test observations)
                if existing_samples is not None:
                    current_samples = np.concatenate([existing_samples, batch_samples], axis=1)
                else:
                    current_samples = batch_samples
                print(f"Current samples shape: {current_samples.shape}")
                
                # Save updated samples to in-progress file
                np.save(fn_samples_test_pred_inprogress, current_samples)
                
                # Update counts
                samples_completed += batch_size
                remaining_samples -= batch_size
                existing_samples = current_samples
                
                # Save simple text checkpoint
                with open(checkpoint_file, 'w') as f:
                    f.write(str(samples_completed))
                
                print(f"Batch completed in {batch_end - batch_start:.2f}s ({(batch_end - batch_start) / 60:.2f} min) ({(batch_end - batch_start) / 3600:.2f} hrs")
                print(f"Saved {samples_completed}/{samples_total} samples")
                
        except Exception as e:
            print(f"Error during sampling: {e}")
            print(f"Partial results saved: {samples_completed}/{samples_total} samples")
            print(f"Resume by running again - will continue from {samples_completed} samples")
            print(f"In-progress file: {fn_samples_test_pred_inprogress}")
            raise
        
        end_time = time.time()
        print(f"Total sampling time (n_samples={n_samples} per obs): {end_time - start_time:.2f}s = {(end_time - start_time) / 60:.2f} min")
        
        # Move in-progress file to final file when complete
        if os.path.exists(fn_samples_test_pred_inprogress):
            os.rename(fn_samples_test_pred_inprogress, fn_samples_test_pred)
            print(f"Sampling complete! Moved to final file: {fn_samples_test_pred}")