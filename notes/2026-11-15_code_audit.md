# muchisimocks code audit — 2026-11-15

Scope: the production pipeline behind the paper figures (notebooks/2026-09-10_paper_figures.ipynb):
data generation, statistics, inference, data loading/ensembles, plotter_figs.py, plotter.py,
utils_plot.py, compute_paper_quantities.py. "[verified]" = re-checked directly against code/data.

Training fields (`_p5_n10000`) do NOT need regenerating. Everything below is either a test-set
regeneration, a statistics recompute, a retrain, or a plotting/paper fix.

---------------------------------------------------------------------------------------------------
## HIGH — can change main results

### H1. Physical-space covariance drops the sigma8–b1 correlation (Figs 8, 9, 10, 11) [verified]
- `utils_inference.py:458-466` (`scale_covariance_unreparameterize_approx`), called from
  `plotter_figs.py:673` (`compute_fob` via `unreparameterize_prediction_block`, utils_inference.py:402),
  `plotter_figs.py:708` (`compute_fom_key_block`), and `load_coverage_metrics`.
- Network infers y = sigma8*b1. Converting to b1 = y/sigma8, the code sets Var(b1)=Var(y)/s8^2 and
  Cov(b1,s8)=0, Cov(b1,om)=Cov(y,om)/s8. Correct linearization:
    Var(b1)     = [Var(y) - 2 b1 Cov(y,s8) + b1^2 Var(s8)] / s8^2
    Cov(b1,s8)  = [Cov(y,s8) - b1 Var(s8)] / s8
    Cov(b1,om)  = [Cov(y,om) - b1 Cov(s8,om)] / s8
  True sigma8–b1 correlation is ~ -0.98, so this is a large error.
- SHAMe nbar=2.2e-4, code vs exact (cov of per-draw unreparameterized samples):
    FoM3 Pgg 7243 vs 10180; Pgg+B 12509 vs 23608
    FoB3 Pgg 1.61 vs 1.99; Pgg+B 2.62 vs 1.68; Pgg+Pgm 1.23 vs 1.87; all 1.43 vs 1.75
  -> Fig 8 ordering of Pgg+B vs Pgg+Pgm flips. FoM(b1) 13–50% too high.
- Point estimate also mean(y)/mean(s8) instead of mean(y/s8) (`utils_inference.py:411`; Fig 4).
- compute_paper_quantities already uses the exact covariance (`:488-516`), so paper FoB numbers
  currently disagree with figures (b1: 0.37 vs 0.43 sigma).
- Unaffected: FoM/FoB of sigma8 and omega alone; contour plots.
- FIX: unreparameterize samples per draw, then mean + np.cov. Rebuild caches (OVERWRITE_CACHE=True).

### H2. Fiducial kp0.35 ensembles use the OLD sweep's hyperparameters [verified]
- `ensemble_kmax_utils.py:383-402`. The fiducial `..._best-rand30{,_nbest1,_nbest2}` members match the
  no-kp-cut sweep's top-3 (zlf6r7gl, ng1yr257, rjjncomd), not the kp0.35 sweep's best_runs.txt
  (41aghyzl, xo44pkec, v9zij2h0). Same for all 4 fiducial combos, pk_bispec kb0.17/kb0.3/no-kb,
  pk_pgm kpgm0.25/0.3/0.35. Most other kmax grid points use the kp0.35 sweep.
- Fig 8/9 fiducial cell differs from neighbours in hyperparameter source, not just k-cut.
- .yaml configs now name the kp0.35 sweep -> retraining from them would silently change member 0.
- Paper text describes the sweep at kmax_Pgg=0.37, Pgm=B=0.25 — doesn't match what was used.

### H3. Validation split leaks across bias draws of the same cosmology [verified]
- `sbi_model.py:790-809`: sbi's internal `validation_fraction` = random row permutation. With bx=32,
  each val row shares its N-body field with ~31 train rows.
- Early stopping, sweep objective (best_validation_loss), top-3 selection all optimistic; worst at
  small N_cosmo / large bx — exactly the "bias sets ≈ cosmologies" convergence result.
- Related: `ensemble_k_selection.py:110-121` scores K on a "held-out" 10% that is ~90% training data;
  its cache is keyed only by sweep root.
- FIX: split by cosmology index, pass val set explicitly.

---------------------------------------------------------------------------------------------------
## MEDIUM

### M1. Test sets reuse training IC seeds [verified] — BEING FIXED (v2 test sets, 2026-11-15)
- `data_creation_pipeline.py:70,74` seed = idx_mock for lh and fixed -> coverage/CV/shame mock i has the
  same phases as training mock i. Noise seeds were offset (0/20000/30000), IC seeds were not.
- Fix in place: SEED_BASE_IC registry (coveragev2 base 100000, shamev2 base 200000) + seed.json.

### M2. P_gm "matter" field is the advected delta operator (known open issue)
- `compute_statistics.py` pgm branch uses `bias_terms_eul[1]`; SHAMe uses N-body matter. ~31% off at
  k=0.35. The "1" operator (`bias_terms_eul[0]`) matches N-body P_mm to ~1% up to k=0.38.
- Likely explains the Fig 8 "Pgm small scales only biased with B" result. Statistics recompute only.

### M3. Noise model cannot produce the constant (1/nbar^2) bispectrum shot-noise term
- `utils_model.get_tracer_field`: m=1 Gaussian eps -> <eps^3>=0. SHAMe B not shot-noise subtracted.
- SHAMe minus phase-matched mock ~ constant 2–3.5e7 vs 1/nbar^2 = 2.07e7 (nbar 2.2e-4).
- Options: m=2 term (eps^2-1) (also gives P_gm noise), free constant B-noise parameter, or consistent
  Poisson subtraction. Paper currently says the noise model is "valid for B_ggg".

### M4. Shared noise field across all 32 bias draws per cosmology
- `data_loader.get_fns_statistic_precomputed`: idxs_noise = [idx_mock]*len(idxs_bias).
- FIX (planned): on-the-fly eps with rng([scheme, set_id, idx_mock, idx_bias]); same eps for pk/bispec/pgm
  of the same row.

### M5. SHAMe bias "truth" questionable
- `utils_model.py:36-41`: b2, bs2 = probabilistic-bias values ×2 (paper quotes 0.127, -0.339).
- Field-level fit on phase-matched mock (code's operator basis): b2≈0.012, bs2≈-0.071, bl≈-0.16
  (vs 0.127, -0.339, -0.82); b1 0.460 vs 0.474. Dict-bias P_gg is 5–13% above SHAMe.
- Affects: SHAMe b2/bs2 "biases" in the paper; CV set truth; noisebest selection.

### M6. Fig 10 FoB3 reference line
- `plotter_figs.py:1370-1371, 1485-1494`: mean FoB3 / sqrt(chi2_3(0.683))=1.877, line at 1. Calibrated
  posterior gives E[sqrt(chi2_3)]/1.877 = 0.85. Fig 11 uses sqrt(2/pi) (expectation). Inconsistent.
- Figs 10/11 use nanmean over obs; paper captions say median [verified, plotter_figs.py:1365-1371].

### M7. choose_best_run resume can skip better-ranked untested runs
- `choose_best_run.py:751-760, 806-819`: counts passes anywhere in ranked list; nbest1/nbest2 may not
  be true 2nd/3rd. Also the pass/fail filter can use the SHAMe mock (mild OOD test leak).

### M8. nth_best_run test configs: wrong mask + filename collisions
- `generate_config_inference.py:126-160`: applies sweep checkpoint to data with config's tags_mask
  (313/374 mismatched); filenames collide across masks with overwrite=False.

### M9. Sweep name mismatch between sweep and best/test modes
- `generate_config_inference.py:300` vs `:307`: matches_sweep_model never matches for kp0.35/kpgm0.25
  -> always retrains.

### M10. Duplicate k-cuts (bin-centre masking, `k < kmax`) [verified]
- kp0.35≡kp0.37 (31 bins), kp0.3≡kp0.32, kpgm0.3≡kpgm0.32, kpgm0.35≡kpgm0.37, kb0.32≡kb0.35 (125).
- Paper says kmax_Pgg=0.37, code says _kp0.35 — same bins (top edge 0.357). Pick one name.

---------------------------------------------------------------------------------------------------
## LOW — latent traps, robustness, labels

Data loading / caching
- No cross-statistic row-alignment assert (`data_loader.py:46-66`); idxs_params from last stat only.
- Cached pgm/bispec masks not size-checked (`get_Pgm_mask`, `get_bispec_mask`); stale mask_pgm_kb0.25.txt.
- `load_or_build` caches keyed by name only; never invalidated.
- Silent partial data: load_fiducial_ensemble_samples_list, Fig 9 report_missing=False,
  _load_ensemble_coverage_3d accepts partial ensembles; NaN rows make FoB/FoM and MSE use different obs.
- `load_coverage_metrics` falls back to nth=0 for non-coverage tags; reports ensemble×N regardless.
- `load_data_shame` cwd-relative path (`data_loader.py:96`); choose_best_run --n-samples-shame defaults 0.
- dead/fragile: 'fisher' in str(int index) (`data_loader.py:997`); 'p0' substring tests; pklin
  precomputed branch unbound.

Training / inference
- "Restore best weights" ineffective (`sbi_model.py:813-834`; train() returns copy first). Harmless now.
- No torch seeds anywhere.
- `_rp` silently drops h, omega_b, ns (`run_inference.py:226-258`); not in tag.
- Resume with different --n-samples -> partly-NaN obs (`sbi_model.py:1254`).
- `scaler_custom.py:75` in-place `x += const`; `scale_error` 'none' returns x.
- Prior float64 vs data float32; sbi exclude_invalid_x silently drops rows.

Statistics
- Noise-only mode crashes (`compute_statistics.py:395-416`); `:364` swallows FileNotFoundError.
- pklin seed for fisher sets mismatched (`:316`).
- process_mock_catalogs: relative paths, stale tracer caches, hard-coded tag_mock, phase-averaged
  Gaussian error not /sqrt(2), assert message uses nonexistent sim.BoxSize.
- Lowest k-bins: few modes, 1000 vs 1024 box effective-k differs slightly.

Figures / labels
- `load_shame_fob3` hidden state in function attributes (set only by Fig 8, `plotter_figs.py:2532`).
- Fig 7 noiseless contour = single model; others K=3 ensembles.
- Fig 8 (nominal cut) vs Fig 9 (upper bin edge) x conventions share KMAX_XLIM.
- FOB3/FOM3 labels say Omega_m; param is omega_cold (cdm+baryons); FOB_LABELS say Omega_c.
- `utils_plot.append_sigma8_cold_x_b1` fills only diagonal of new column.
- `plotter.plot_contours_inf` mutable default extents={}.
- `filter_finite_coverage_rows` dead and broken.

Generation
- `cosmo_shame_p5_n1000` registry would let bacco override varied omega/sigma8 (generate_params.py:164-171).
- Fisher mode builds wrong params filename (`data_creation_pipeline.py:99`).
- Re-run can leave ~5 GB intermediates per mock (`:179-192` vs `:307-312`).
- os.system return codes unchecked; stale param copies in mock dirs.
- generate_noise_fields.py has no coverage entry (actual fields used base 20000).
- omega_m - neutrino_mass subtracts eV from a density (inactive, m_nu=0).
- generate_params builds a bacco Cosmology at import.
- k-cut Nyquist plane not Hermitian-consistent (only k≈0.40, above all cuts).

---------------------------------------------------------------------------------------------------
## Paper text vs code
- Figs 10/11 "median" vs code mean.
- FoB eq: 1.88 = sqrt(chi2_3(0.68)), not chi2_3(0.68)=3.53.
- kmax_Pgg 0.37 vs _kp0.35 (equivalent bins).
- "Omega_c" = omega_cold (cdm+baryons); define once.
- Fig A text "10 random" vs code/caption 5; "1,0000" typo.
- "independent" CV/coverage sets — shared phases (fixed in v2).
- Sweep description vs actual hyperparameter source (H2).
- Noise model "valid for B_ggg" (M3).
