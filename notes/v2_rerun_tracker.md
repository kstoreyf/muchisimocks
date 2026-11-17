# v2 rerun tracker

Fixes from the 2026-11-15 audit (`notes/2026-11-15_code_audit.md`), in rerun order. IDs refer to that report.
Status: `[ ]` to do · `[~]` in progress · `[x]` done · `[d]` decided / no change · `[-]` won't fix
Outputs: `source code_auxiliary/env_v2.sh` → `/scratch/kstoreyf/muchisimocks/{data_v2,results_v2}`.

## Phase 1 — Code fixes before recomputing statistics
- [x] **M4 Independent noise per bias draw** — `compute_statistics.py`: tag_noise == `_noisefly` → ε from `default_rng([2, crc32(tag_params), idx_mock, idx_bias])`, generated on the fly; same ε for pk/bispec/pgm of a row. Tag: `_noisefly` (seed uses tag_params, so no set suffix needed). Tested: deterministic, row/set corr ~0.001
- [x] **M2 P_gm matter field** — `compute_statistics.py` uses `bias_terms_eul[0]`; `utils_model.pnn_to_pgm` indices → [0,1,2,3,4] (checked vs direct, ratio 1.0000). New/old P_gm: ~1 at large scales, 0.70–0.76 at k=0.38
- [ ] **M3 Bispectrum shot-noise term** — decide: m=2 (ε²−1) term / free constant B noise / Poisson subtraction
- [ ] **Bispectrum binning** — leaning linear Δk≈0.035 (≈5.6 k_F); confirm with Marcos what binning his validation paper uses (unverified), since the kb0.25 cut leans on it
- [x] **H3 Validation split by cosmology** — `sbi_model.NPEGroupedVal` + `grouped_val_indices` (whole cosmologies held out, fixed seed 42); `run_inference` passes `groups_train=idxs_params[idxs_train,0]`. Toy test: 0 cosmologies shared. Also fixed best-weights restore (load into the returned copy)
- [ ] `ensemble_k_selection.py:110-121` K-selection still scores on training-overlapping rows
- [x] **Torch seeds** for training (`seed` config key, default 0). Sampling not seeded (multiprocess workers; 10k draws)
- [x] **H1 Covariance unreparameterization** — new `ui.physical_moments_from_samples` (exact); used by load_shame_fob3, load_ensemble_fom3/_marg, load_coverage_metrics, Fig 4. Moments-only map now full Jacobian (J C Jᵀ), within ~2% of exact. Cache dir → `figures/paper_figures_cache_v2`. Reproduces exact SHAMe FoB3 1.99/1.87/1.68/1.75
- [ ] Row-alignment assert across statistics (`data_loader.py:46-66`); size checks on cached pgm/bispec masks

## Phase 2 — Test sets and SHAMe truths
- [x] **M1 v2 test sets with independent IC phases** — `_coveragev2_p5_n1000` (base 100000), `_cvshamev2_p0_n1000` (base 200000; SHAMe-cosmology cosmic-variance set, renamed from `_shamev2_p0_n1000` on 2026-11-17: lib dir, seed.json, params file, SEED_BASE_IC). Jobs 7973620 / 7973621, all 1000 + 1000 mocks done
- [ ] Sync v2 test libs scratch → `/data/kstoreyf/muchisimocks/`
- [x] **M5 SHAMe bias truth** — ÷2 on b2, bs2 confirmed with colleague (2026-11-15); `bias_dict_shame` now stores raw/2; fixed bias param files rewritten
- [x] Redo noisebest selection with ÷2 biases (fit 7982679; matched stats 7982680-2; matched test set rebuilt; 11-12 nb updated)

## Phase 3 — Recompute statistics (v2 tree)
- [~] Training `_p5_n10000` × `_biasnoisenestv2_p9_n320000` × `_noisefly` (v2 noise priors: An_homog ±5, An_b1 ±5, An_b2 ±3, An_bs2 ±5, An_bl ±20): params job 7975563 → pgm 7975564 → pk 7975565 (aftercorr); bispec 7975566 (200 × 50 cosmologies, linear binning). The earlier `_biasnoisenest` pk/pgm (old priors) are superseded.
- [~] Coverage v2: pk, pgm, bispec. pgm 7996092 → pk 7996093 (aftercorr; `_coveragev2_p5_n1000` × `_biasnoisecoveragev2_p9_n1000` × `_noisefly`, 100 tasks × 10 mocks, submitted 2026-11-16 20:35), done 1000/1000. Bispec 7999329 (`bispec_v2_cov`, submitted 2026-11-17)
- [~] CV v2 (cvshamev2): pk, pgm, bispec. `_cvshamev2_p0_n1000` × `_biasshame_noisebest_p0_n1` (nbar 2.2e-4) × `_noisefly`: pgm 7999316 → pk 7999317 (aftercorr), bispec 7999318 (100 tasks × 10 mocks, submitted 2026-11-17)
- [ ] SHAMe mock P_gm/B consistent with any noise-model / shot-noise choice
- [ ] Spot-check: v2 vs old statistics (pk should be ~unchanged in mean; pgm shifted)

## Phase 4 — Hyperparameter sweep and model selection
- [ ] **H2 One consistent sweep source** for every retrained model; record which in the paper
- [ ] **M7** `choose_best_run` resume logic; drop SHAMe from pass/fail filter (or disclose)
- [ ] **M8/M9** config generation: mask mismatch, filename collisions, sweep-name mismatch
- [ ] K-selection on genuinely held-out cosmologies
- [ ] **Training/sweep plan (agreed 2026-11-16)**, to implement on `cleanup-v2`:
  - [ ] **Batch size fixed at 128**, removed from the sweep grid. In v1, batch 32–256 showed no trend, so a fixed size keeps one training setup across the convergence grid. 128 is about 1.7× faster than 64 (benchmark below), and the smallest convergence models (500 rows) still get about 4 steps per epoch.
  - [ ] **Sweep grid:** learning rate {1e-3, 3e-4, 1e-4, 3e-5} × hidden {128, 256, 512} × transforms {6, 8, 10}, which is 36 combinations. rand16 covers 44% of them.
  - [ ] **Training jobs:** 1 core (more threads are slower, see `code_auxiliary/bench_train_threads.py`), regular queue, 24 h.
  - [ ] **Resume after timeout** (sbi `resume_training`), so slow trials (hidden 512 with learning rate 3e-5, about 525 epochs fit in 24 h) continue in a follow-up job instead of being lost.
  - [ ] **Ensembles = the best sweep setting retrained with 3 seeds**, replacing top-3 sweep trials. In the v1 rand30 sweeps the 3rd-best trial was 0.2–0.7 nats of validation loss worse than the best (0.73 for full P+B+P_gm), so the v1 top-3 ensembles mixed in noticeably worse models, and going to rand20 barely helps. Identical-settings ensembles follow Alvey, Contaldi & Pieroni 2025 (arXiv:2507.13495), which makes the spread between members pure seed noise.
  - [ ] **Ensemble disagreement diagnostic (Alvey+25):** compute the pairwise KL between seed members on the SHAMe mocks and compare with in-distribution test sets, as a misspecification check. Optional: pairwise KL vs N_train in the convergence grid (expected to fall like 1/N), but that needs seed ensembles at every N, about 3× the convergence-grid training.
  - Benchmark (CPU, one thread, minutes per 320k-row epoch, batch 64 → 128): hidden 128: 1.30 → 0.76; hidden 256: 2.07 → 1.24; hidden 512: 5.15 → 2.74. Batch 1024 is another ~2× faster but is untested for quality. Node speed varies by about 25%.

## Phase 5 — k_max scan and fiducial cuts
- [ ] Rerun Fig 8/9 grids with fixed FoB/FoM; re-choose kb / kpgm fiducial cuts
- [ ] **M10** pick one name for the P_gg cut (0.35 vs 0.37); define kmax as bin-centre cut

## Phase 6 — Train fiducial ensembles and convergence grid
- [ ] Fiducial K=3 ensembles (4 stat combos): best setting × 3 seeds (see Phase 4 plan)
- [ ] Noiseless model for Fig 7 as an ensemble too
- [ ] Convergence grid (N_cosmo × N_bias)

## Phase 7 — Figures and paper
- [ ] Rebuild all caches (`OVERWRITE_CACHE=True`, v2 cache dir)
- [ ] **M6** Fig 10 FoB3 reference (calibrated = 0.85) ; mean vs median consistent with captions
- [ ] Labels: Ω_m vs ω_cold, FoB normalization in y-label, k units
- [ ] Paper text: FoB eq (√χ²), "independent" test sets, sweep description, noise model "valid for B", Fig A 5 vs 10, typos
- [ ] compute_paper_quantities numbers consistent with figures

## Phase 8 — Robustness and cleanup (low)
- [ ] `load_shame_fob3` hidden state; silent partial-data fallbacks; `load_or_build` cache keys
- [ ] "Restore best weights" (`sbi_model.py:813-834`); in-place scaler; resume NaN rows
- [ ] Relative paths (`load_data_shame`, `process_mock_catalogs`), unchecked `os.system`
- [ ] Remaining LOW items in the audit report

## Log
- 2026-11-15 09:31: home EDQUOT again (even `touch` failed; beegfs-ctl showed 35.6/50 GB, 221k/300k files) → all jobs died writing logs. Checked 75,471 pgm/pk files since 09:25 and intermediates of 10 half-built coverage mocks: all OK. Resubmitted as 7973620/21 (datagen), 7973622 (pgm) → 7973623 (pk). Job logs now go to `/scratch/kstoreyf/muchisimocks/logs_v2/` (set in both job scripts).
- 2026-11-15 09:12: pgm array 7973255 hit 3G limit (peak ~3.14G; 9 of first 13 tasks OOM-killed) → aftercorr pk tasks stuck DependencyNeverSatisfied. Cancelled 7973255/7973256, `job_stats_v2.s` → `--mem=6G`, resubmitted full arrays as 7973411 (pgm) / 7973412 (pk). Checked 77,768 pgm+pk files written since 08:53: one truncated (`pgm_5413_b295413.npy`), deleted for recompute.
- 2026-11-15: H1 covariance fix, H3 grouped val split, torch seeds, best-weights restore. Coverage-metrics path checked on real data (full combo FoB3/1.877 = 0.855, i.e. calibrated).
- 2026-11-15 08:21: cluster /home outage killed all jobs (exit 120 / 0:53). Checked: 40,688 pgm files OK (newest 2,877 all load); removed 9 half-built coverage mocks; resubmitted from code_auxiliary/ (logs in code_auxiliary/logs/). Resume skips finished outputs.
- 2026-11-15: per-row noise + P_gm fix in compute_statistics.py; also loads each cosmology's fields once (was per row). Training pgm→pk arrays submitted (100 tasks, %40). Script: code_auxiliary/jobs/job_stats_v2.s
- 2026-11-15: audit done; v2 conventions set (`env_v2.sh`, `SEED_BASE_IC`, seed.json); v2 test-set datagen submitted (20 GPUs). First coverage mock OK (seed 100000).
- 2026-11-15: renamed on-the-fly noise tag `_noise_unit_perrow{tag_params}` → `_noisefly` (code + job script); mv'd the two finished training dirs (pks/pgms, 320k files each).
- 2026-11-15 evening: SHAMe bias convention → ÷2 (was ×2). `select_shame_noisebest.py` settings now 'fixed b' / 'free b'; noisebest re-picked with fixed ÷2 biases (2.2e-4). `params_fixed_biasshame_noisem2best_p0_n1.txt` still ×2 (stale, m2 model only).
- 2026-11-15: noise priors widened from the phase-matched fits (notebook §2.5); new param set `_biasnoisenestv2_p9_n320000`; noisebest re-selected on the matched mock (`code_auxiliary/select_shame_noisebest.py`, params_fixed_biasshame_noisebest_p0_n1 overwritten).

## Phase-matched muchisimocks test set (`_shamematched_p0_n10`), set up 2026-11-15, not run yet
**Other densities (2026-11-17):** `_shamematched{nbar}_p0_n10` × `_biasshame{nbar}_noisebest_p0_n1` for nbar 0.00011, 0.00054 (truth = `bias_dict_shame['_nbar{nbar}']` + that density's fixed-b noise fit, `select_shame_noisebest.py write`). Stats 7999662 / 7999663 (`matched_noisebest_nbar*`) → test-set write 7999666. `params_fixed_biasshame0.00022_noisebest_p0_n1.txt` also written (identical to `_biasshame_noisebest_p0_n1`); renaming the 2.2e-4 tags to the nbar form is pending a decision (touches the v1 pipeline + ~1700 result files).
SHAMe Rings ICs populated like the CV set (bias_dict_shame + noisebest), 10 noise realizations; obs = 0/π pair mean; P_gm vs the '1' operator; B rescaled to the loader's n_grid=128 convention. Truth: `_biasshame_noisebest_p0_n1`.
1. ~~`sbatch jobs/job_shame_matched_noisebest.s`~~ submitted 7982680 (after the ÷2 noisebest fit)
2. `source env_v2.sh && python make_shame_matched_testset.py`  (seconds; writes data_v2 test-set dirs + params_fixed_shamematched_p0_n10.txt)
3. once the models exist: `sbatch --array=0-5 jobs/job_train_v2.s jobs/lists/noise_ab_v2_matched.txt`  (test only: individual + `_mean` for the 6 with/without-noise models)
