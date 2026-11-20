# Pipeline: order of steps (broad strokes)

How a full rerun goes, from parameters to paper figures, with the decisions that gate each stage. Specifics (job ids, status) live in `notes/v2_rerun_tracker.md`. Run everything through sbatch; the login node is only for quick checks.

## 0. Setup
- `source code_auxiliary/env_v2.sh` points data/results at a fresh tree (`/scratch/.../{data_v2,results_v2}`), so the old products stay intact.
- Fix the code first (audit → tracker Phase 1). Recomputing statistics is the expensive step, so get noise/P_gm/bispectrum choices settled before it.

## 1. Parameters and test sets
1. Training parameters: cosmology LH (`_p5_n10000`) × bias+noise draws (`_biasnoisenest..._p9_n320000`) × noise realization tag (`_noisefly`). `generate_params.py`.
2. Test sets, each with IC phases independent of training:
   - coverage set (`_coveragev2_p5_n1000` × `_biasnoisecoveragev2_p9_n1000`);
   - cosmic-variance set at the SHAMe cosmology (`_cvshamev2_p0_n1000`);
   - SHAMe mocks at each n̄ (fixed bias/noise truth, `_biasshame{nbar}_noisebest_p0_n1`);
   - phase-matched mocks (SHAMe ICs + our forward model, `_shamematched{nbar}_p0_n10`) at each n̄.
3. SHAMe truths: bias conventions (÷2 on b2, bs2), then noise "best" fit per n̄ (`select_shame_noisebest.py` / `compute_shame_matched_*`).
4. Coverage subsets for quick checks: `select_coverage_subset.py` (`inner500`, `inner500_rand100`).

## 2. Statistics
- `compute_statistics.py` per set: P_gm and P (chained, aftercorr), bispectrum separately (slowest). Training first, then coverage / CV / matched.
- k-masks: `make_masks.py`, on the shared K_CUTS grid (cut labelled by top bin edge).
- Spot-check against the previous version before training on it.

## 3. Interim models (optional, while the sweep runs)
- Train the previous version's best hyperparameters on the new data at the fiducial cuts (`make_v1hp_configs.py`). Use them to make early method decisions (e.g. noise in θ or not, seeds vs hyperparameter sets) and to check the test machinery end to end.

## 4. Hyperparameter sweep
- One W&B sweep per statistics combination (P, P+P_gm, P+B, P+P_gm+B) at the fiducial cuts: `code/submit_scripts/submit_inf_train_sweep.sh` (24 h, checkpoints + self-requeue).
- `choose_best_run.py` → `best_runs.txt` (ranked trials).
- Train extra seeds of the top setting (`make_best_seeds_list_v2.py`). Seed scatter tells you whether the trial ranking is meaningful.

## 5. Choose the ensemble
- For each combo compare top-K settings (seed 0) vs K seeds of the top setting: coverage on `inner500` + SHAMe contours (`make_ensemble_tests_rand16.py`, 11-19 v2_checks notebook §2).
- Pick one scheme for the paper. The top setting seed 0 is in either ensemble, so its long tests (full 1000-row coverage, CV mean, matched) can start before the choice.

## 6. Choose the fiducial k_max
- Retrain the top setting (best mode re-uses its hyperparameters at any other mask) over a k_max grid: fix kp, vary kpgm and kb over K_CUTS (`make_kmax_grid_rand16.py`).
- Test on SHAMe ×3 n̄ and phase-matched ×3 n̄ (individual + mean). Pick the largest cuts where SHAMe stays unbiased relative to the matched mocks.
- If the fiducial k_max changes, steps 4–5 were run at the wrong cuts. Decide whether to re-sweep or transfer hyperparameters.

## 7. Production models and tests at the fiducial choice
- Ensemble members at the fiducial cuts, tested on: full coverage (1000), CV set (mean + individual), SHAMe ×3 n̄, matched ×3 n̄.
- k_max sweeps of the ensemble for the FoB vs k_max figure.
- Convergence grid (box fraction `bx` × number of cosmologies), only after the fiducial k_max is fixed.
- Noiseless variant (needs its own training data) for the noiseless CV-mean comparison.

## 8. Paper figures
- `plotter_figs.py` + the paper figures notebook (latest `notebooks/*_paper_figures_v2.ipynb`). Check stat-combo naming matches the model dirs (e.g. `pk_pgm_bispec`).

## Operational notes
- Job lists: `job_train_v2.s <list>` with one `<train config|none> <test configs...>` line per array task. Training and tests resume after interruption; tests retry on transient /scicomp failures.
- Slurm snapshots the job script at submit. Edits don't reach already-queued jobs, but code edits do, so don't change `code/` under running jobs without thinking.
- Sampling must run single-threaded (`torch.set_num_threads(1)` in `run_inference.py`, since the bacco import resets OMP threads).
- Give each job a descriptive `--job-name`. Logs go to `/scratch/.../logs_v2`.
- Notebook updates: execute via `jobs/job_nbexec.s`, then install the executed copy atomically.
