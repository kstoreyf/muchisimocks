import multiprocessing as mp
import os
import queue
import re
import shutil
from types import SimpleNamespace

import numpy as np
import pathlib
import math
import pickle
import time
import tempfile
import yaml
import wandb

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
import torch
from torch.distributions import biject_to, constraints
from sbi.utils import BoxUniform
from sbi.utils.sbiutils import within_support

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
_SWEEP_LOCK_WAIT_S = 900.0  # > stale time, so waiters outlast a dead holder's lock


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


# --- Resumable training: checkpoint every few epochs, stop cleanly before a job deadline -------
# fit_model trains in chunks of TRAIN_CHUNK_EPOCHS and pickles the sbi inference object (network,
# optimizer, epoch, best-val state, train/val split) to <out_dir>/TRAIN_CHECKPOINT after each one.
# If MUCHISIMOCKS_TRAIN_DEADLINE (unix seconds) would be passed by another chunk, it stops and
# sets SBIModel.training_incomplete; run_inference then exits with EXIT_TRAINING_INCOMPLETE and
# the job requeues itself. A rerun with the same out_dir continues from the checkpoint; in sweep
# mode the stopped W&B run (state failed/crashed + checkpoint) is resumed under the same run id.
TRAIN_CHUNK_EPOCHS = 5
TRAIN_CHECKPOINT = "checkpoint_inference.p"
TRAIN_INCOMPLETE_MARKER = "training_incomplete.txt"
TRAIN_CLAIM = "resume_claim.txt"
TRAIN_MAX_RESUMES = 20
EXIT_TRAINING_INCOMPLETE = 3
_RESUMABLE_STATES = {"crashed", "failed", "killed", "preempted", "preempting"}


def _train_deadline():
    raw = os.environ.get("MUCHISIMOCKS_TRAIN_DEADLINE", "").strip()
    return float(raw) if raw else None


def _atomic_pickle(obj, fn):
    fn_tmp = f"{fn}.tmp{os.getpid()}"
    with open(fn_tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(fn_tmp, fn)


def _job_id():
    jid = os.environ.get("SLURM_ARRAY_JOB_ID")
    if jid and os.environ.get("SLURM_ARRAY_TASK_ID"):
        return f"{jid}_{os.environ['SLURM_ARRAY_TASK_ID']}"
    return os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")


def _claim_is_live(fn_claim):
    """A resume claim is live if it names another Slurm job that is still queued/running."""
    try:
        owner = pathlib.Path(fn_claim).read_text().strip()
    except OSError:
        return False
    if not owner or owner == _job_id() or owner.startswith("pid"):
        return False
    try:
        import subprocess
        out = subprocess.run(["squeue", "-h", "-j", owner, "-o", "%T"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return True  # can't tell: don't steal it
    return bool(out)


def _is_resumable_run_dir(run_dir):
    """Run dir holds a training checkpoint, isn't claimed by a live job, and has resumes left."""
    run_dir = pathlib.Path(run_dir)
    if not (run_dir / TRAIN_CHECKPOINT).is_file() or (run_dir / "inference.p").is_file():
        return False
    if _claim_is_live(run_dir / TRAIN_CLAIM):
        return False
    try:
        with open(run_dir / TRAIN_CHECKPOINT, "rb") as f:
            n_resumes = int(pickle.load(f).get("n_resumes", 0))
    except Exception as e:
        print("checkpoint %s unreadable (%s); not resumable" % (run_dir, e), flush=True)
        return False
    return n_resumes < TRAIN_MAX_RESUMES


def _claim_resumable_run(sweep_api_path, dir_sbi):
    """Pick a stopped sweep run with a usable checkpoint and claim it for this job (call under the
    sweep lock). Returns the run id or None."""
    api = wandb.Api()
    sweep = api.sweep(sweep_api_path)
    for run in sweep.runs:
        if str(getattr(run, "state", "")).lower() not in _RESUMABLE_STATES:
            continue
        run_dir = pathlib.Path(dir_sbi) / str(run.id)
        if _is_resumable_run_dir(run_dir):
            (run_dir / TRAIN_CLAIM).write_text(_job_id() + "\n")
            return str(run.id)
    return None


def _retire_crashed_runs_in_sweep(sweep_api_path, dir_sbi):
    """
    Record, then remove from the sweep, runs that crashed/failed/were killed WITHOUT a training
    checkpoint (those with one are resumed instead), so the sweep can draw a replacement (crashed
    runs count against ``run_cap``).

    Each one is appended to ``dir_sbi/killed_trials.csv`` (hyperparameters, state, runtime) and its
    local run dir moved to ``dir_sbi/killed_trials/<run_id>/``, so trials that were too slow to
    finish stay visible instead of silently dropping out of the sweep.
    """
    api = wandb.Api()
    sweep = api.sweep(sweep_api_path)
    root = pathlib.Path(dir_sbi)
    to_remove = [
        r
        for r in sweep.runs
        if str(getattr(r, "state", "")).lower() in _RESUMABLE_STATES
        # a stopped run with a checkpoint is resumed, not retired (unless claimed by a live job,
        # in which case it is being resumed right now)
        and not (root / str(r.id) / TRAIN_CHECKPOINT).is_file()
    ]
    if not to_remove:
        return
    killed_dir = root / "killed_trials"
    fn_log = root / "killed_trials.csv"
    cols = ["run_id", "state", "created_at", "runtime_h", *_WANDB_SWEEP_PARAMETER_KEYS, "sweep"]
    root.mkdir(parents=True, exist_ok=True)
    new_file = not fn_log.is_file()
    n_retired = 0
    with open(fn_log, "a", encoding="utf-8") as f:
        if new_file:
            f.write(",".join(cols) + "\n")
        for run in to_remove:
            rid = str(run.id)
            runtime = run.summary.get("_runtime")
            row = [rid, str(run.state), str(getattr(run, "created_at", "")),
                   "" if runtime is None else "%.2f" % (float(runtime) / 3600.0),
                   *[str(run.config.get(k, "")) for k in _WANDB_SWEEP_PARAMETER_KEYS],
                   sweep_api_path]
            f.write(",".join(row) + "\n")
            f.flush()
            p = root / rid
            if _WANDB_RUN_ID_DIR_RE.match(rid) and p.is_dir():
                killed_dir.mkdir(exist_ok=True)
                shutil.move(str(p), str(killed_dir / rid))
            try:
                run.delete()
            except Exception as e:
                print("wandb: could not delete crashed run %s: %s" % (rid, e), flush=True)
                continue
            n_retired += 1
            print("wandb: retired %s run %s (%s) -> %s" % (run.state, rid, dict(
                (k, run.config.get(k)) for k in _WANDB_SWEEP_PARAMETER_KEYS), fn_log), flush=True)
    print("wandb: retired %d crashed/failed/killed runs from %s (logged in %s)"
          % (n_retired, sweep_api_path, fn_log), flush=True)


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


VAL_SPLIT_SEED = 42  # fixed so validation losses are comparable across sweep runs / ensemble members


def grouped_val_indices(groups, validation_fraction, seed=VAL_SPLIT_SEED):
    """Row indices of a validation set made of whole groups (cosmologies).

    With bx bias/noise draws per cosmology, a random row split puts the same N-body field in
    both train and val, so early stopping and the sweep objective can't see overfitting to a
    field's cosmic variance. Holding out whole cosmologies avoids that.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(validation_fraction * len(uniq))))
    val_groups = rng.choice(uniq, size=n_val, replace=False)
    return np.nonzero(np.isin(groups, val_groups))[0]


class NPEGroupedVal(NPE):
    """NPE whose train/val split is fixed in advance (e.g. by cosmology) instead of random rows."""

    def __init__(self, *args, val_indices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_val_indices = None if val_indices is None else np.asarray(val_indices)

    def get_dataloaders(self, starting_round=0, training_batch_size=200,
                        validation_fraction=0.1, resume_training=False, dataloader_kwargs=None):
        if self._fixed_val_indices is not None and not resume_training:
            n = self.get_simulations(starting_round)[0].size(0)
            is_val = np.zeros(n, dtype=bool)
            is_val[self._fixed_val_indices] = True
            self.train_indices = torch.from_numpy(np.nonzero(~is_val)[0])
            self.val_indices = torch.from_numpy(np.nonzero(is_val)[0])
            # resume_training=True makes sbi reuse the indices set above (it only affects the split)
            resume_training = True
        return super().get_dataloaders(starting_round, training_batch_size, validation_fraction,
                                       resume_training, dataloader_kwargs)


# ---------------------------------------------------------------------------
# Prior support and rejection sampling.
#
# With ``reparameterize: true`` the flow is trained on sigma8*b (and sigma8**2*b) columns. The
# bounding box of those product columns is much larger than their true support (a wedge: for a
# given sigma8, sigma8*b1 lies in [sigma8*lo, sigma8*hi]). sbi only rejects draws outside the
# prior's ``support``, so with a plain BoxUniform the empty corners were kept and mapped back to
# physical bias values outside the prior. ``PhysicalSupportBoxUniform`` keeps the same box for
# training but its support check tests physical bounds.
#
# ``sample_batched_isolated`` replaces ``DirectPosterior.sample_batched``, whose rejection loop
# keeps drawing for every observation until the worst one is done (one leaky observation
# stalls the whole batch). NOTE: pickled posteriors reference PhysicalSupportBoxUniform /
# PhysicalBoxConstraint by module path (sbi_model) — don't move or rename them.
# ---------------------------------------------------------------------------

_PREFIX_SQ = utils_inference.REPARAM_PREFIX_SQ
_PREFIX_X = utils_inference.REPARAM_PREFIX_X
_SIGMA8 = "sigma8_cold"


def physical_param_name(name: str) -> tuple[str, int]:
    """(physical name, sigma8 power) for a training column name; power 0 if not reparameterized."""
    if name.startswith(_PREFIX_SQ):
        return name[len(_PREFIX_SQ):], 2
    if name.startswith(_PREFIX_X):
        return name[len(_PREFIX_X):], 1
    return name, 0


class PhysicalBoxConstraint(constraints.Constraint):
    """Support = physical prior box, evaluated on (possibly reparameterized) training coords."""

    is_discrete = False
    event_dim = 1

    def __init__(self, param_names, dict_bounds_physical, box_low, box_high):
        # box_* = bounding box in training coords; only used for sbi's MCMC/MAP bijection.
        self.box_low = torch.as_tensor(box_low, dtype=torch.float32)
        self.box_high = torch.as_tensor(box_high, dtype=torch.float32)
        names = [str(n) for n in param_names]
        phys, powers = zip(*(physical_param_name(n) for n in names))
        missing = [p for p in phys if p not in dict_bounds_physical]
        if missing:
            raise KeyError(f"dict_bounds_physical missing bounds for {missing}")
        if any(powers) and _SIGMA8 not in names:
            raise ValueError("reparameterized columns present but sigma8_cold is not a column")
        self.param_names = names
        self.idx_sigma8 = names.index(_SIGMA8) if _SIGMA8 in names else None
        self.powers = torch.tensor(powers, dtype=torch.float32)
        self.low = torch.tensor([dict_bounds_physical[p][0] for p in phys], dtype=torch.float32)
        self.high = torch.tensor([dict_bounds_physical[p][1] for p in phys], dtype=torch.float32)

    def to_physical(self, value: torch.Tensor) -> torch.Tensor:
        if self.idx_sigma8 is None:
            return value
        s8 = value[..., self.idx_sigma8 : self.idx_sigma8 + 1]
        # sigma8 <= 0 gives inf/nan here; those draws fail the bounds test below (and sigma8
        # itself is checked against its own bounds).
        return value / s8 ** self.powers.to(value.device)

    def check(self, value: torch.Tensor) -> torch.Tensor:
        phys = self.to_physical(value)
        low, high = self.low.to(value.device), self.high.to(value.device)
        return ((phys >= low) & (phys <= high)).all(dim=-1)


@biject_to.register(PhysicalBoxConstraint)
def _biject_physical_box(constraint):
    # sbi's build_posterior builds an unconstrained-space transform from prior.support (used
    # for MCMC / MAP only). The bounding-box bijection is valid for that purpose.
    return biject_to(
        constraints.independent(constraints.interval(constraint.box_low, constraint.box_high), 1)
    )


class PhysicalSupportBoxUniform(BoxUniform):
    """BoxUniform over the training-coordinate bounding box whose *support* is the physical box.

    Only the support matters for single-round NPE (training loss does not use the prior;
    sampling and posterior leakage correction use ``support.check``). ``log_prob`` is the box
    constant inside the true support and -inf outside; it is not the exact prior density in
    reparameterized coordinates (that carries a sigma8 Jacobian), and ``sample`` draws from the
    bounding box — neither is used by our pipeline.
    """

    def __init__(self, low, high, param_names, dict_bounds_physical, **kwargs):
        super().__init__(low=low, high=high, **kwargs)
        self._physical_support = PhysicalBoxConstraint(
            param_names, dict_bounds_physical, box_low=low, box_high=high
        )
        self._log_box_density = float(-torch.log(torch.as_tensor(high) - torch.as_tensor(low)).sum())

    @property
    def support(self):
        return self._physical_support

    def log_prob(self, value):
        inside = self._physical_support.check(value)
        return torch.where(
            inside,
            torch.full(inside.shape, self._log_box_density, dtype=value.dtype, device=value.device),
            torch.full(inside.shape, -math.inf, dtype=value.dtype, device=value.device),
        )


def sample_batched_isolated(
    posterior,
    x,
    n_samples: int,
    *,
    max_draws_per_round: int = 2_000_000,
    min_acceptance: float = 1e-3,
    verbose: bool = True,
):
    """Rejection-sample ``n_samples`` draws per observation from an sbi ``DirectPosterior``.

    All still-active observations are drawn in one vectorized call per round (same speed as
    sbi's ``sample_batched`` for well-behaved batches), but finished observations drop out, so
    later rounds spend the whole draw budget on the few leaky ones. An observation is
    abandoned (all-NaN) once it has used ``n_samples / min_acceptance`` draws, i.e. once its
    acceptance is known to be below ``min_acceptance``; other observations are unaffected.

    Returns
    -------
    samples : torch.Tensor, shape (n_samples, n_obs, n_params) — same layout as sbi.
    acceptance : np.ndarray, shape (n_obs,) — accepted / drawn per observation.
    """
    estimator = posterior.posterior_estimator
    prior = posterior.prior
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    n_obs = x.shape[0]
    max_draws_per_obs = int(math.ceil(n_samples / min_acceptance))

    accepted = [[] for _ in range(n_obs)]
    n_acc = torch.zeros(n_obs, dtype=torch.long)
    n_drawn = torch.zeros(n_obs, dtype=torch.long)
    active = torch.arange(n_obs)
    abandoned = []
    n_rounds = 0
    t0 = time.time()

    while len(active) > 0:
        n_rounds += 1
        need = (n_samples - n_acc[active]).float()
        rate = torch.where(
            n_drawn[active] > 0,
            n_acc[active].float() / n_drawn[active].clamp(min=1).float(),
            torch.ones(len(active)),
        ).clamp(min=min_acceptance)
        m_needed = int(torch.ceil(1.2 * need / rate).max().item())
        m = max(100, min(m_needed, max_draws_per_round // len(active)))

        with torch.no_grad():
            cand = estimator.sample((m,), condition=x[active])  # (m, n_active, d)
        ok = within_support(prior, cand).reshape(m, len(active))

        for j, i in enumerate(active.tolist()):
            if n_acc[i] < n_samples:
                a = cand[ok[:, j], j]
                accepted[i].append(a)
                n_acc[i] += a.shape[0]
        n_drawn[active] += m

        done = n_acc[active] >= n_samples
        give_up = ~done & (n_drawn[active] >= max_draws_per_obs)
        abandoned.extend(active[give_up].tolist())
        active = active[~done & ~give_up]

    n_params = cand.shape[-1]
    samples = torch.full((n_samples, n_obs, n_params), float("nan"))
    for i in range(n_obs):
        if i in abandoned:
            continue
        samples[:, i] = torch.cat(accepted[i], dim=0)[:n_samples]
    acceptance = (n_acc.float() / n_drawn.clamp(min=1).float()).numpy()

    if verbose:
        print(
            f"sample_batched_isolated: {n_obs} obs x {n_samples} samples in {n_rounds} rounds, "
            f"{time.time() - t0:.1f}s; acceptance min/median = "
            f"{acceptance.min():.3g}/{np.median(acceptance):.3g}",
            flush=True,
        )
        if abandoned:
            print(
                f"sample_batched_isolated: WARNING abandoned {len(abandoned)} obs with acceptance "
                f"< {min_acceptance:g} (NaN): idx {abandoned[:20]}"
                f"{' ...' if len(abandoned) > 20 else ''}",
                flush=True,
            )
    return samples, acceptance


class SBIModel():

    def __init__(self, theta_train=None, y_train_unscaled=None,
                     theta_test=None, y_test_unscaled=None,
                     statistics=None, param_names=None, dict_bounds=None, dict_bounds_physical=None,
                     run_mode='single', tag_sbi='', n_threads=1, 
                     sweep_name=None, overwrite=False,
                     # True iff bx/n_train match BX_SWEEP/N_TRAIN_SWEEP and tags_mask matches
                     # tags_mask_for_sweep(statistics); run_mode best: copy sweep artifact vs retrain.
                     matches_sweep_model=False,
                     wandb_sweep_id=None,
                     sweep_num_runs=None,
                     wandb_config_yaml_path=None,
                     nth_best_run=None,
                     sweep_run_id=None,
                     groups_train=None,
                     seed=0,
                     ):

        self.dir_sbi = str(paths.DIR_RESULTS / "results_sbi" / f"sbi{tag_sbi}")
        self.tag_sbi = tag_sbi
        p = pathlib.Path(self.dir_sbi)
        p.mkdir(parents=True, exist_ok=True)
        
        self.theta_train = theta_train
        self.y_train_unscaled = y_train_unscaled
        # cosmology index per training row; if given, validation holds out whole cosmologies
        self.groups_train = None if groups_train is None else np.asarray(groups_train)
        self.seed = int(seed)

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
            assert dict_bounds_physical is not None, 'need dict_bounds_physical if training model'
        self.dict_bounds = dict_bounds
        # physical (un-reparameterized) prior bounds; defines the posterior support
        self.dict_bounds_physical = dict_bounds_physical
        self.last_acceptance = None
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
        # Explicit sweep trial id (any rank, e.g. to retrain a mid-ranked trial); overrides nth_best_run
        self.sweep_run_id = sweep_run_id
        assert sweep_run_id is None or nth_best_run is None, "set only one of sweep_run_id / nth_best_run"

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
        
        project_name = os.environ.get("MUCHISIMOCKS_WANDB_PROJECT", "muchisimocks-sbi")
        self.training_incomplete = False
        
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
                    # Pruned from 14 rand30 sweeps (Nov 2026): lr 1e-5/3e-3/1e-2, hidden 32/64 and
                    # 4 transforms never came within 0.3 nats of a sweep's best; 256 was the usual
                    # winner (top edge), so 512 is added. Batch size showed no trend; kept in full.
                    'learning_rate': {'values': [1e-3, 3e-4, 1e-4, 3e-5]},
                    'hidden_features': {'values': [128, 256, 512]},
                    'training_batch_size': {'values': [32, 64, 128, 256]},
                    'num_transforms': {'values': [6, 8, 10]},
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
                    _retire_crashed_runs_in_sweep(sweep_api_path, self.dir_sbi)
                    resume_run_id = _claim_resumable_run(sweep_api_path, self.dir_sbi)
                    print(
                        "wandb sweep resume: path=%s target_total_finished_runs=%s"
                        % (sweep_api_path, self.sweep_num_runs),
                        flush=True,
                    )
                else:
                    resume_run_id = None
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

            if resume_run_id is not None:
                # Continue a trial that stopped before its job deadline (or was killed) from its
                # checkpoint, under the same W&B run id, instead of drawing a new trial.
                print("wandb: resuming stopped sweep run %s from its checkpoint" % resume_run_id,
                      flush=True)
                run_dir = os.path.join(self.dir_sbi, resume_run_id)
                wandb.init(project=agent_project, entity=agent_entity, id=resume_run_id,
                           resume="must")
                self._fit_and_log_wandb_trial(wandb.config, run_dir)
                return

            n_finished = _wandb_count_finished_success_runs(sweep_api_path)
            n_total = len(wandb.Api().sweep(sweep_api_path).runs)
            if n_total >= int(self.sweep_num_runs):
                print("wandb: sweep already has %d/%d runs (%d finished) and none to resume; "
                      "nothing to do" % (n_total, self.sweep_num_runs, n_finished), flush=True)
                return
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
                self._fit_and_log_wandb_trial(wandb.config, os.path.join(self.dir_sbi, wandb.run.id))

            wandb.agent(
                sweep_id,
                function=_fit_model_wandb,
                count=count,
                entity=agent_entity,
                project=agent_project,
            )
            wandb.finish()
            
        elif self.run_mode == 'best':
            # matches_sweep_model True: copy the trained sweep trial into dir_sbi. False: retrain
            # with that trial's hyperparameters (its config.pkl) on this config's data.
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
            if self.sweep_run_id is not None:
                best_run_id = str(self.sweep_run_id)
                print("run_mode=best: sweep_run_id=%r (explicit trial)" % best_run_id, flush=True)
            elif self.nth_best_run is not None:
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

            # Every sweep trial is saved locally under the sweep dir (fit_model save_dir), so the
            # chosen trial must be there; there is no W&B download fallback.
            if not os.path.isfile(local_cfg_pkl):
                raise SystemExit(
                    "run_mode=best: sweep trial %s has no %s; the sweep dir must hold every trial"
                    % (best_run_id, local_cfg_pkl)
                )
            if self.matches_sweep_model:
                # Same data as the sweep: reuse the trained trial.
                if not os.path.isfile(local_posterior):
                    raise SystemExit("run_mode=best: missing trained trial %s" % local_posterior)
                os.makedirs(self.dir_sbi, exist_ok=True)
                for fn in ["posterior.p", "inference.p", "param_names.txt", "config.pkl"]:
                    shutil.copy2(os.path.join(local_run_dir, fn), os.path.join(self.dir_sbi, fn))
                for f in pathlib.Path(local_run_dir).glob("scaler_y_*.p"):
                    shutil.copy2(f, os.path.join(self.dir_sbi, f.name))
                print(
                    "run_mode=best: copied sweep trial %s (same bx/n_train/tags_mask) to %s"
                    % (best_run_id, self.dir_sbi),
                    flush=True,
                )
            else:
                # Different data: retrain with the trial's hyperparameters.
                try:
                    cfg = _fit_config_from_config_pkl(local_cfg_pkl, max_epochs, validation_fraction)
                except ValueError as e:
                    raise SystemExit(str(e)) from e
                print(
                    "run_mode=best: retraining with hparams of trial %s (%s) into %s"
                    % (best_run_id, local_cfg_pkl, self.dir_sbi),
                    flush=True,
                )
                wandb.login()
                wandb.init(
                    project=project_name,
                    config=vars(cfg),
                    name="retrain-besthp-%s" % best_run_id,
                )
                self.fit_model(cfg, save_model=True)
                wandb.finish(exit_code=EXIT_TRAINING_INCOMPLETE if self.training_incomplete else 0)

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
            wandb.finish(exit_code=EXIT_TRAINING_INCOMPLETE if self.training_incomplete else 0)

        elif self.run_mode == 'load':
            self.load_posterior()
            self.load_param_names()
        else:
            raise ValueError(f"run_mode {self.run_mode} not recognized")
            
    def _fit_and_log_wandb_trial(self, config, run_dir):
        """Train one sweep trial into ``run_dir`` (resuming its checkpoint if there is one). If it
        stops before the job deadline, end the W&B run as failed (exit code
        EXIT_TRAINING_INCOMPLETE) so the next job resumes it; else log the model artifact."""
        try:
            self.fit_model(config, save_model=True, save_dir=run_dir)
        finally:
            fn_claim = os.path.join(run_dir, TRAIN_CLAIM)
            if os.path.exists(fn_claim):
                os.remove(fn_claim)
        if self.training_incomplete:
            wandb.finish(exit_code=EXIT_TRAINING_INCOMPLETE)
            return
        artifact = wandb.Artifact(name=f"sbi-posterior-{wandb.run.id}", type="model")
        artifact.add_dir(local_path=run_dir)
        wandb.run.log_artifact(artifact)
        wandb.finish()

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
        # Box over training coords for sbi, but support = physical prior box: with sigma8
        # reparameterization the product-column bounding box is mostly outside the true
        # support (see PhysicalSupportBoxUniform).
        prior = PhysicalSupportBoxUniform(
            low=torch.from_numpy(l_bounds),
            high=torch.from_numpy(u_bounds),
            param_names=self.param_names,
            dict_bounds_physical=self.dict_bounds_physical,
        )

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

        out_dir = save_dir if save_dir is not None else self.dir_sbi
        os.makedirs(out_dir, exist_ok=True)
        fn_ckpt = f"{out_dir}/{TRAIN_CHECKPOINT}"
        config_dict = _training_config_to_plain_dict(config)
        self.training_incomplete = False
        ckpt = None
        if os.path.exists(fn_ckpt):
            with open(fn_ckpt, "rb") as f:
                ckpt = pickle.load(f)
            hp_ckpt = {k: ckpt["config"].get(k) for k in _WANDB_SWEEP_PARAMETER_KEYS}
            hp_now = {k: config_dict.get(k) for k in _WANDB_SWEEP_PARAMETER_KEYS}
            assert hp_ckpt == hp_now, f"checkpoint {fn_ckpt} has hyperparameters {hp_ckpt}, not {hp_now}"
            assert int(ckpt.get("seed", 0)) == self.seed, f"checkpoint {fn_ckpt} seed != {self.seed}"
            print(f"Resuming training from {fn_ckpt}: epoch {ckpt['inference'].epoch}, "
                  f"resume #{ckpt['n_resumes'] + 1}", flush=True)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if ckpt is not None:
            inference = ckpt["inference"]
        elif self.groups_train is not None:
            assert len(self.groups_train) == len(self.theta_train), "groups_train must match theta_train rows"
            val_indices = grouped_val_indices(self.groups_train, validation_fraction)
            print(f"Validation split by cosmology: {len(np.unique(self.groups_train[val_indices]))} of "
                  f"{len(np.unique(self.groups_train))} cosmologies, {len(val_indices)} rows (seed {VAL_SPLIT_SEED})")
            inference = NPEGroupedVal(prior=prior, density_estimator=density_estimator_build_fun,
                                      val_indices=val_indices)
        else:
            print("WARNING: no groups_train given; sbi splits validation by random rows "
                  "(rows of one cosmology can land in both train and val)")
            inference = NPE(prior=prior, density_estimator=density_estimator_build_fun)
        if ckpt is None:
            inference = inference.append_simulations(
                torch.tensor(self.theta_train, dtype=torch.float32),
                torch.tensor(self.y_train, dtype=torch.float32),
                )

        print(f"Training with {self.n_threads} threads")
        start = time.time()

        # Train in chunks; checkpoint after each; stop before the job deadline (see TRAIN_CHECKPOINT).
        deadline = _train_deadline()
        n_resumes = 0 if ckpt is None else int(ckpt["n_resumes"]) + 1
        resume = ckpt is not None
        while True:
            t_chunk = time.time()
            epoch_target = min(max_epochs, (inference.epoch if resume else 0) + TRAIN_CHUNK_EPOCHS)
            density_estimator = inference.train(
                max_num_epochs=epoch_target,
                training_batch_size=training_batch_size,
                validation_fraction=validation_fraction,
                learning_rate=learning_rate,
                show_train_summary=False,
                resume_training=resume,
            )
            resume = True
            # train() exits with epoch == epoch_target + 1 unless it converged (early stopping)
            done = inference.epoch <= epoch_target or epoch_target >= max_epochs
            dt_chunk = time.time() - t_chunk
            print(f"epoch {inference.epoch}: val loss {inference._val_loss:.4f} (best "
                  f"{inference._best_val_loss:.4f}, {inference._epochs_since_last_improvement} since "
                  f"improvement), chunk {dt_chunk / 60:.1f} min", flush=True)
            if done:
                break
            _atomic_pickle({"inference": inference, "config": config_dict, "seed": self.seed,
                            "n_resumes": n_resumes}, fn_ckpt)
            if deadline is not None and time.time() + 1.5 * dt_chunk + 600 > deadline:
                with open(f"{out_dir}/{TRAIN_INCOMPLETE_MARKER}", "w") as f:
                    f.write(f"stopped at epoch {inference.epoch} before deadline {deadline:.0f}\n")
                print(f"Stopping before the job deadline at epoch {inference.epoch}; checkpoint "
                      f"{fn_ckpt} (rerun to resume)", flush=True)
                self.training_incomplete = True
                return
        # sbi reloads best-val weights inside the training loop only when early stopping
        # triggers; if we stop because max_num_epochs is hit first, weights can still be
        # from the last epoch. Always restore the best checkpoint (no-op if already loaded).
        best_sd = getattr(inference, "_best_model_state_dict", None)
        if best_sd is not None and inference._neural_net is not None:
            inference._neural_net.load_state_dict(best_sd)
            # train() returns a deepcopy made before this reload; build_posterior uses that copy
            density_estimator.load_state_dict(best_sd)
        for fn in (fn_ckpt, f"{out_dir}/{TRAIN_INCOMPLETE_MARKER}", f"{out_dir}/{TRAIN_CLAIM}"):
            if os.path.exists(fn):
                os.remove(fn)
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
            
            print(f"min and max after scaling (func={func_scaler_y}): {np.min(y_train_i):3f}, {np.max(y_train_i):3f}")
            
            # save scaler - need pickle for custom object!!
            # atomic: parallel sweep agents rewrite these while finished trials copy them
            fn_scaler_y = f'{self.dir_sbi}/scaler_y_{statistic}.p'
            fn_tmp = f'{fn_scaler_y}.tmp{os.getpid()}'
            with open(fn_tmp, "wb") as f:
                pickle.dump(scaler_y, f)
            os.replace(fn_tmp, fn_scaler_y)
                
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
        
        # Per-observation early exit: one leaky observation can no longer stall the batch.
        samples, self.last_acceptance = sample_batched_isolated(
            self.posterior, y_obs, n_samples
        )
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
        Returns (samples, acceptance, timed_out); acceptance is per obs (NaN if timed out).
        """
        if batch_timeout_seconds is None or batch_timeout_seconds <= 0:
            samples = self.evaluate(y_test_unscaled_batch, n_samples=n_samples)
            return samples, self.last_acceptance, False

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
                    np.full(batch_size, np.nan),
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
            samples, acceptance = payload
            return samples, acceptance, False
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
        # Per-obs fraction of flow draws inside the physical prior (NaN = not sampled / timed
        # out / abandoned below min_acceptance). Low values flag leaky posteriors.
        fn_acceptance = f"{self.dir_sbi}/samples_test{tag_test_eval}_acceptance.npy"

        if y_test_unscaled[0].ndim == 1:
            samples_total = 1
        else:
            samples_total = len(y_test_unscaled[0])
        acceptance = np.full(samples_total, np.nan)
        if not self.overwrite and os.path.exists(fn_acceptance):
            prev = np.load(fn_acceptance)
            if prev.shape == acceptance.shape:
                acceptance = prev
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
        n_draws = int(n_samples)
        if existing_samples is not None and existing_samples.shape[0] != n_draws:
            # Resizing would truncate finished obs or pad them with NaN rows that still
            # count as "usable"; make the caller choose (rerun with overwrite, or same n_samples).
            raise ValueError(
                f"Existing samples for {tag_test_eval!r} have {existing_samples.shape[0]} draws "
                f"per obs but n_samples={n_draws}; rerun with the same n_samples or overwrite."
            )
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
                batch_samples, batch_acceptance, timed_out = self._evaluate_batch_with_timeout(
                    y_test_unscaled_batch,
                    n_samples=n_samples,
                    batch_size=batch_size,
                    batch_timeout_seconds=batch_timeout_seconds,
                )
                batch_end = time.time()

                self._write_batch_into_canvas(canvas, batch_indices, batch_samples)
                acceptance[np.asarray(batch_indices)] = np.asarray(batch_acceptance, dtype=float)
                np.save(fn_acceptance, acceptance)

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
        q.put(("ok", (samples, np.asarray(model.last_acceptance))))
    except Exception as ex:
        q.put(("error", repr(ex)))