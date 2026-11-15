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
- [~] **M1 v2 test sets with independent IC phases** — `_coveragev2_p5_n1000` (base 100000), `_shamev2_p0_n1000` (base 200000). Jobs 7973620 / 7973621 (resubmitted 09:45, logs in /scratch/kstoreyf/muchisimocks/logs_v2/); ~11 min/mock
- [ ] Sync v2 test libs scratch → `/data/kstoreyf/muchisimocks/`
- [ ] **M5 SHAMe bias truth** — field-level fit on phase-matched mock vs `bias_dict_shame` (×2 convention, bl)
- [ ] Rebuild CV set bias params + redo noisebest selection with the chosen biases and noise model

## Phase 3 — Recompute statistics (v2 tree)
- [~] Training `_p5_n10000` × `_biasnoisenest_p9_n320000` × `_noisefly`: pgm (job 7973622) → pk (7973623, aftercorr) at `--mem=6G`, resubmitted 09:45; bispec still to do (~1,400–2,000 core-h)
- [ ] Coverage v2: pk, pgm, bispec
- [ ] CV v2 (shamev2): pk, pgm, bispec
- [ ] SHAMe mock P_gm/B consistent with any noise-model / shot-noise choice
- [ ] Spot-check: v2 vs old statistics (pk should be ~unchanged in mean; pgm shifted)

## Phase 4 — Hyperparameter sweep and model selection
- [ ] **H2 One consistent sweep source** for every retrained model; record which in the paper
- [ ] **M7** `choose_best_run` resume logic; drop SHAMe from pass/fail filter (or disclose)
- [ ] **M8/M9** config generation: mask mismatch, filename collisions, sweep-name mismatch
- [ ] K-selection on genuinely held-out cosmologies

## Phase 5 — k_max scan and fiducial cuts
- [ ] Rerun Fig 8/9 grids with fixed FoB/FoM; re-choose kb / kpgm fiducial cuts
- [ ] **M10** pick one name for the P_gg cut (0.35 vs 0.37); define kmax as bin-centre cut

## Phase 6 — Train fiducial ensembles and convergence grid
- [ ] Fiducial K=3 ensembles (4 stat combos)
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
