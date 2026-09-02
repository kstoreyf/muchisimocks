import multiprocessing as mp
import os
import queue
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


_SWEEP_LOCK_STALE_S = 300.0
_SWEEP_LOCK_WAIT_S = 180.0


def _env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y"}


def _env_int(name, default=None):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _read_wandb_sweep_id_from_yaml(yaml_path):
    p = pathlib.Path(yaml_path)
    if not p.is_file():
        return ""
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return ""
    return str(data.get("wandb_sweep_id") or "").strip()


def _sweep_lock_dir(yaml_path):
    p = pathlib.Path(yaml_path)
    return p.with_name(p.name + ".sweep.lock")


def _acquire_sweep_lock(lock_dir):
    """Cross-node lock via atomic mkdir (configs live on a shared filesystem)."""
    lock_dir = pathlib.Path(lock_dir)
    t0 = time.time()
    while True:
        try:
            lock_dir.mkdir()
            return
        except FileExistsError:
            try:
                age = time.time() - lock_dir.stat().st_mtime
            except OSError:
                continue
            if age > _SWEEP_LOCK_STALE_S:
                print(
                    "wandb: stealing stale sweep lock %s (age=%.0fs)"
                    % (lock_dir, age),
                    flush=True,
                )
                try:
                    lock_dir.rmdir()
                except OSError:
                    pass
                continue
            if time.time() - t0 > _SWEEP_LOCK_WAIT_S:
                raise TimeoutError("Timed out waiting for sweep lock %s" % lock_dir)
            time.sleep(0.5)


def _release_sweep_lock(lock_dir):
    try:
        pathlib.Path(lock_dir).rmdir()
    except OSError as e:
        print("wandb: could not remove sweep lock %s: %s" % (lock_dir, e), flush=True)


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
        try:
            run.delete()
        except Exception as e:
            print("wandb: could not delete crashed run %s: %s" % (rid, e), flush=True)
            continue
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
                     nth_best_run=None,
                     ):

        self.dir_sbi = str(paths.DIR_RESULTS / "results_sbi" / f"sbi{tag_sbi}")
        self.tag_sbi = tag_sbi
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
        self.nth_best_run = None if nth_best_run is None else int(nth_best_run)

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
            sweep_parallel = _env_flag("MUCHISIMOCKS_SWEEP_PARALLEL")
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
            lock_dir = (
                _sweep_lock_dir(self.wandb_config_yaml_path)
                if self.wandb_config_yaml_path
                else None
            )
            if lock_dir is not None:
                _acquire_sweep_lock(lock_dir)
            try:
                if self.wandb_config_yaml_path:
                    yaml_id = _read_wandb_sweep_id_from_yaml(self.wandb_config_yaml_path)
                    if yaml_id:
                        resume_id = yaml_id
                        self.wandb_sweep_id = yaml_id
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
            finally:
                if lock_dir is not None:
                    _release_sweep_lock(lock_dir)

            if sweep_parallel:
                print(
                    "wandb: skipping orphaned local dir cleanup (parallel sweep agents)",
                    flush=True,
                )
            else:
                _delete_orphaned_local_sweep_run_dirs(self.dir_sbi, sweep_api_path)

            n_finished = _wandb_count_finished_success_runs(sweep_api_path)
            remaining = max(0, int(self.sweep_num_runs) - n_finished)
            per_agent = _env_int("MUCHISIMOCKS_SWEEP_RUNS_PER_AGENT")
            if per_agent is not None:
                count = min(remaining, max(0, per_agent))
            else:
                count = remaining
            print(
                "wandb: agent count=%s (target finished runs=%s, already finished=%s, "
                "runs_per_agent=%s, parallel=%s)"
                % (
                    count,
                    self.sweep_num_runs,
                    n_finished,
                    per_agent if per_agent is not None else "all remaining",
                    sweep_parallel,
                ),
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
                "run_mode=best: looking for best run id in sweep dir %s "
                "(output dir_sbi=%s, sweep_name=%r, nth_best_run=%s)"
                % (sweep_dir, self.dir_sbi, self.sweep_name, self.nth_best_run),
                flush=True,
            )
            if self.nth_best_run is not None:
                from generate_config_inference import nth_passing_run_id
                try:
                    best_run_id = nth_passing_run_id(
                        pathlib.Path(sweep_dir), int(self.nth_best_run)
                    )
                except (FileNotFoundError, IndexError, ValueError) as e:
                    raise SystemExit(str(e)) from e
                print(
                    "run_mode=best: nth_best_run=%s → W&B run id %r from %s/best_runs.txt"
                    % (self.nth_best_run, best_run_id, sweep_dir),
                    flush=True,
                )
            else:
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
    
    
    
    @staticmethod
    def _n_test_obs_in_samples(samples_arr):
        """Test observations saved along axis 1: (n_draws, n_obs, n_params) or (n_draws, n_params)."""
        if samples_arr.ndim == 3:
            return samples_arr.shape[1]
        if samples_arr.ndim == 2:
            return 1
        raise ValueError(
            f"Unexpected samples array shape {samples_arr.shape}; expected 2D or 3D"
        )

    @staticmethod
    def _obs_is_usable(samples_arr, obs_idx=0):
        """True if this observation has any finite posterior draw (failed batches are all-NaN)."""
        if samples_arr is None:
            return False
        if samples_arr.ndim == 2:
            return bool(np.any(np.isfinite(samples_arr)))
        if samples_arr.ndim == 3:
            if obs_idx < 0 or obs_idx >= samples_arr.shape[1]:
                return False
            return bool(np.any(np.isfinite(samples_arr[:, obs_idx, :])))
        raise ValueError(
            f"Unexpected samples array shape {samples_arr.shape}; expected 2D or 3D"
        )

    @staticmethod
    def samples_array_fully_usable(samples_arr, samples_total=None):
        """
        True if ``samples_arr`` covers ``samples_total`` obs (or all stored obs when
        ``samples_total`` is None) with no all-NaN placeholders.
        """
        if samples_arr is None:
            return False
        n_stored = SBIModel._n_test_obs_in_samples(samples_arr)
        n_need = n_stored if samples_total is None else int(samples_total)
        if n_stored < n_need:
            return False
        if samples_arr.ndim == 2:
            return SBIModel._obs_is_usable(samples_arr)
        return all(SBIModel._obs_is_usable(samples_arr, i) for i in range(n_need))

    @staticmethod
    def pending_obs_indices(samples_arr, samples_total):
        """Obs indices that are missing from the array or currently all-NaN."""
        pending = []
        n_stored = 0 if samples_arr is None else SBIModel._n_test_obs_in_samples(samples_arr)
        for i in range(int(samples_total)):
            if samples_arr is None or i >= n_stored or not SBIModel._obs_is_usable(samples_arr, i):
                pending.append(i)
        return pending

    @staticmethod
    def _nan_batch_samples(n_samples, batch_size, n_params):
        """Placeholder block for a timed-out coverage batch (axis-1 slots stay aligned)."""
        if batch_size == 1:
            return np.full((n_samples, n_params), np.nan, dtype=np.float64)
        return np.full((n_samples, batch_size, n_params), np.nan, dtype=np.float64)

    @staticmethod
    def _ensure_samples_canvas(existing_samples, samples_total, n_draws, n_params):
        """
        Full (n_draws, samples_total, n_params) canvas; copy any existing obs slots.
        Single-obs 2D arrays are expanded to 3D for uniform slot writes.
        """
        canvas = np.full(
            (int(n_draws), int(samples_total), int(n_params)),
            np.nan,
            dtype=np.float64,
        )
        if existing_samples is None:
            return canvas
        if existing_samples.ndim == 2:
            canvas[:, 0, :] = np.asarray(existing_samples, dtype=np.float64)
            return canvas
        n_copy = min(existing_samples.shape[1], int(samples_total))
        canvas[:, :n_copy, :] = np.asarray(existing_samples[:, :n_copy, :], dtype=np.float64)
        return canvas

    @staticmethod
    def _write_batch_into_canvas(canvas, obs_indices, batch_samples):
        """Write a batch result into canvas columns ``obs_indices``."""
        batch_samples = np.asarray(batch_samples)
        if batch_samples.ndim == 2:
            if len(obs_indices) != 1:
                raise ValueError(
                    f"2D batch samples for {len(obs_indices)} obs indices; expected 1"
                )
            canvas[:, obs_indices[0], :] = batch_samples
            return canvas
        if batch_samples.ndim != 3:
            raise ValueError(f"Unexpected batch samples shape {batch_samples.shape}")
        if batch_samples.shape[1] != len(obs_indices):
            raise ValueError(
                f"Batch width {batch_samples.shape[1]} != len(obs_indices)={len(obs_indices)}"
            )
        for j, obs_idx in enumerate(obs_indices):
            canvas[:, obs_idx, :] = batch_samples[:, j, :]
        return canvas

    def _evaluate_batch_with_timeout(
        self,
        y_test_unscaled_batch,
        *,
        n_samples,
        batch_size,
        batch_timeout_seconds,
    ):
        """
        Run posterior sampling for one batch in a subprocess so a stall can be killed.
        Returns (samples, timed_out).
        """
        if batch_timeout_seconds is None or batch_timeout_seconds <= 0:
            return self.evaluate(y_test_unscaled_batch, n_samples=n_samples), False

        ctx = mp.get_context("spawn")
        q = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_evaluate_test_batch_worker,
            args=(
                q,
                self.tag_sbi,
                list(self.statistics),
                np.asarray(self.param_names, dtype=str),
                y_test_unscaled_batch,
                int(n_samples),
                bool(self.overwrite),
            ),
        )
        proc.start()
        try:
            # Get before join: a large q.put() blocks until the parent reads, so
            # join-then-get deadlocks once the payload exceeds the pipe buffer.
            try:
                status, payload = q.get(timeout=float(batch_timeout_seconds))
            except queue.Empty:
                proc.terminate()
                proc.join(timeout=10.0)
                print(
                    f"Batch timed out after {batch_timeout_seconds:.0f}s — "
                    f"filling {batch_size} observation(s) with NaN and continuing"
                )
                return (
                    self._nan_batch_samples(
                        n_samples, batch_size, len(self.param_names),
                    ),
                    True,
                )
            proc.join(timeout=10.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
            if proc.exitcode not in (0, None):
                raise RuntimeError(
                    f"Batch worker exited with code {proc.exitcode}"
                )
            if status != "ok":
                raise RuntimeError(f"Batch worker failed: {payload}")
            return payload, False
        finally:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)

    def evaluate_test_set(self, y_test_unscaled=None, tag_test_eval='',
                          n_samples=10000, checkpoint_every=10,
                          resume=True, n_test_eval=100,
                          batch_timeout_seconds=None):
        
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
        fn_failed_indices = f"{self.dir_sbi}/failed_obs_indices_test{tag_test_eval}.txt"

        if y_test_unscaled[0].ndim == 1:
            samples_total = 1
        else:
            samples_total = len(y_test_unscaled[0])
        existing_samples = None
        
        print(f"Checkpoint file: {checkpoint_file}")
        
        if self.overwrite:
            print("Overwrite is True - starting fresh")
            existing_samples = None
            for fn in (
                fn_samples_test_pred,
                fn_samples_test_pred_inprogress,
                checkpoint_file,
                fn_failed_indices,
            ):
                if os.path.exists(fn):
                    os.remove(fn)
        elif resume:
            # Prefer in-progress (active run); else final pred (may have NaN holes to retry).
            if os.path.exists(fn_samples_test_pred_inprogress):
                existing_samples = np.load(fn_samples_test_pred_inprogress)
                print(
                    f"Found existing in-progress samples file with "
                    f"{self._n_test_obs_in_samples(existing_samples)}/{samples_total} "
                    f"stored obs (array shape {existing_samples.shape})"
                )
            elif os.path.exists(fn_samples_test_pred):
                existing_samples = np.load(fn_samples_test_pred)
                print(
                    f"Found existing samples file: {fn_samples_test_pred} "
                    f"(array shape {existing_samples.shape})"
                )

            if existing_samples is not None and self.samples_array_fully_usable(
                existing_samples, samples_total=samples_total,
            ):
                print(
                    f"All {samples_total} test observations already complete "
                    f"(no NaN placeholders) — nothing to do."
                )
                # Ensure final filename exists for callers that only look for _pred.npy.
                if not os.path.exists(fn_samples_test_pred):
                    np.save(fn_samples_test_pred, existing_samples)
                    if os.path.exists(fn_samples_test_pred_inprogress):
                        os.remove(fn_samples_test_pred_inprogress)
                return

        # Build a full-width canvas so we can fill arbitrary NaN / missing slots.
        n_params = len(self.param_names)
        if existing_samples is not None:
            n_draws = existing_samples.shape[0]
        else:
            n_draws = int(n_samples)
        canvas = self._ensure_samples_canvas(
            existing_samples, samples_total, n_draws, n_params,
        )
        pending = self.pending_obs_indices(canvas, samples_total)
        if not pending:
            print(f"All {samples_total} test observations already completed!")
            np.save(fn_samples_test_pred, canvas)
            if os.path.exists(fn_samples_test_pred_inprogress):
                os.remove(fn_samples_test_pred_inprogress)
            return

        n_usable = samples_total - len(pending)
        print(
            f"Pending {len(pending)}/{samples_total} obs "
            f"({n_usable} usable already; will retry NaN/missing slots)"
        )
        # Demote final pred while working so overwrite=False re-entry resumes holes.
        if os.path.exists(fn_samples_test_pred) and not os.path.exists(
            fn_samples_test_pred_inprogress
        ):
            os.rename(fn_samples_test_pred, fn_samples_test_pred_inprogress)
        np.save(fn_samples_test_pred_inprogress, canvas)
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            f.write(str(n_usable))

        print(
            f"Batching: checkpoint_every={checkpoint_every}, "
            f"batch_timeout_seconds={batch_timeout_seconds}"
        )
        
        start_time = time.time()
        # Timed-out slots stay NaN on disk (retryable next job), but must not re-enter
        # ``pending`` after we recompute it from the canvas each iteration.
        skipped_this_run = set()

        try:
            while pending:
                batch_indices = pending[:checkpoint_every]
                batch_size = len(batch_indices)
                print(
                    f"Sampling batch of {batch_size} obs "
                    f"(pending {len(pending)}/{samples_total}; "
                    f"indices {batch_indices[0]}..{batch_indices[-1]})"
                )

                if y_test_unscaled[0].ndim == 1:
                    y_test_unscaled_batch = y_test_unscaled
                else:
                    y_test_unscaled_batch = [
                        y_stat[batch_indices] for y_stat in y_test_unscaled
                    ]

                batch_start = time.time()
                print(f"Evaluating obs indices {batch_indices}")
                batch_samples, timed_out = self._evaluate_batch_with_timeout(
                    y_test_unscaled_batch,
                    n_samples=n_samples,
                    batch_size=batch_size,
                    batch_timeout_seconds=batch_timeout_seconds,
                )
                batch_end = time.time()

                # Worker may return a different n_draws; resize canvas if needed.
                batch_samples = np.asarray(batch_samples)
                n_draws_batch = batch_samples.shape[0]
                if n_draws_batch != canvas.shape[0]:
                    new_canvas = np.full(
                        (n_draws_batch, samples_total, n_params),
                        np.nan,
                        dtype=np.float64,
                    )
                    n_copy = min(canvas.shape[0], n_draws_batch)
                    new_canvas[:n_copy] = canvas[:n_copy]
                    canvas = new_canvas

                self._write_batch_into_canvas(canvas, batch_indices, batch_samples)

                if timed_out:
                    with open(fn_failed_indices, "a", encoding="utf-8") as f:
                        for obs_idx in batch_indices:
                            f.write(f"{obs_idx}\n")
                else:
                    # Drop these indices from the failed list if a prior run marked them.
                    if os.path.exists(fn_failed_indices):
                        prev = [
                            int(line.strip())
                            for line in open(fn_failed_indices, encoding="utf-8")
                            if line.strip()
                        ]
                        batch_set = set(batch_indices)
                        kept = [i for i in prev if i not in batch_set]
                        with open(fn_failed_indices, "w", encoding="utf-8") as f:
                            for i in kept:
                                f.write(f"{i}\n")

                print(f"Batch samples shape: {batch_samples.shape}")
                print(f"Canvas shape: {canvas.shape}")
                np.save(fn_samples_test_pred_inprogress, canvas)

                pending_all = self.pending_obs_indices(canvas, samples_total)
                n_usable = samples_total - len(pending_all)
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    f.write(str(n_usable))

                print(
                    f"Batch completed in {batch_end - batch_start:.2f}s "
                    f"({(batch_end - batch_start) / 60:.2f} min) "
                    f"({(batch_end - batch_start) / 3600:.2f} hrs)"
                    f"{' [TIMED OUT — NaN placeholder]' if timed_out else ''}"
                )
                print(f"Usable {n_usable}/{samples_total} samples "
                      f"({len(pending_all)} still pending on disk)")

                # If this batch timed out, its slots are still NaN and would loop forever
                # if we only dropped the current batch — previously skipped NaNs are
                # reintroduced by pending_obs_indices. Accumulate skips for this run.
                if timed_out:
                    skipped_this_run.update(batch_indices)

                pending = [i for i in pending_all if i not in skipped_this_run]
                if timed_out:
                    if pending:
                        print(
                            f"Skipping {batch_size} timed-out obs for this run "
                            f"({len(skipped_this_run)} skipped total); "
                            f"{len(pending)} other pending remain"
                        )
                    else:
                        print(
                            "No non-timed-out pending obs left in this run; "
                            "re-submit later to retry NaN slots"
                        )
                        break

        except Exception as e:
            n_usable = samples_total - len(self.pending_obs_indices(canvas, samples_total))
            print(f"Error during sampling: {e}")
            print(f"Partial results saved: {n_usable}/{samples_total} usable samples")
            print("Resume by running again — will retry remaining NaN/missing obs")
            print(f"In-progress file: {fn_samples_test_pred_inprogress}")
            raise
        
        end_time = time.time()
        print(
            f"Total sampling time (n_samples={n_samples} per obs): "
            f"{end_time - start_time:.2f}s = {(end_time - start_time) / 60:.2f} min"
        )

        pending = self.pending_obs_indices(canvas, samples_total)
        if not pending:
            if os.path.exists(fn_samples_test_pred_inprogress):
                os.replace(fn_samples_test_pred_inprogress, fn_samples_test_pred)
            else:
                np.save(fn_samples_test_pred, canvas)
            if os.path.exists(fn_failed_indices):
                os.remove(fn_failed_indices)
            print(f"Sampling complete! Final file: {fn_samples_test_pred}")
        else:
            np.save(fn_samples_test_pred_inprogress, canvas)
            print(
                f"Still pending {len(pending)}/{samples_total} obs "
                f"(left as NaN placeholders). Re-run to retry."
            )


def _evaluate_test_batch_worker(
    q,
    tag_sbi,
    statistics,
    param_names,
    y_batch,
    n_samples,
    overwrite,
):
    """Subprocess entry point for one coverage batch (enables hard timeout via terminate)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        model = SBIModel(
            tag_sbi=tag_sbi,
            run_mode="load",
            param_names=param_names,
            statistics=statistics,
            overwrite=overwrite,
        )
        model.run()
        samples = model.evaluate(y_batch, n_samples=n_samples)
        # Convert to numpy before queueing: torch tensors use FD-based
        # resource sharing, which fails with ConnectionRefusedError if the
        # worker exits before the parent finishes unpickling (join-then-get).
        if hasattr(samples, "detach"):
            samples = samples.detach().cpu().numpy()
        else:
            samples = np.asarray(samples)
        q.put(("ok", samples))
    except Exception as ex:
        q.put(("error", repr(ex)))