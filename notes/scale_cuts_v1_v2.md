# Scale cuts: v1 vs v2 labels

The v2 convention is on branch `cleanup-v2` (worktree `~/muchisimocks-cleanup`, commit 7309d49) and is **not yet merged into `main`**. Models trained from `main`, including the 2026-11 noise A/B (`kp0.35_kpgm0.25`), still use v1 labels.

## Rules

- **v1:** `_kp<X>` keeps the bins whose **centre** is below X. A label can therefore admit modes above X: kp0.35 keeps a bin that extends to k = 0.356.
- **v2:** keeps the bins whose **top edge** is at or below X. Labels must come from one grid shared by all statistics, `data_loader.K_CUTS = (0.11, 0.15, 0.18, 0.21, 0.24, 0.27, 0.31, 0.34, 0.37)`. An empty tag means no cut, plotted at 0.40. A label that isn't on the grid raises an error.
- **Bispectrum:** a triangle is kept only if all three k fall in kept bins.
- **Binning** (`data_loader.K_EDGES`, the same as `compute_statistics`):
  - P and P_gm: 32 log bins over [0.01, 0.4], geometric centres.
  - B: 12 linear bins, Δk = 0.0325, 203 triangles in total.
- **Masks:** precomputed by `code_auxiliary/make_masks.py` into `<DIR_DATA>/masks/`. The loader checks them against the data's k.

## v2 label grid: what each cut keeps

| v2 label | P / P_gm bins | top edge | last bin centre | B k-bins | top edge | B triangles |
|---|---|---|---|---|---|---|
| 0.11 | 20 | 0.1003 | 0.0947 | 3 | 0.1075 | 7 |
| 0.15 | 23 | 0.1417 | 0.1338 | 4 | 0.1400 | 13 |
| 0.18 | 25 | 0.1785 | 0.1685 | 5 | 0.1725 | 22 |
| 0.21 | 26 | 0.2003 | 0.1891 | 6 | 0.2050 | 34 |
| 0.24 | 27 | 0.2248 | 0.2122 | 7 | 0.2375 | 50 |
| 0.27 | 28 | 0.2522 | 0.2381 | 8 | 0.2700 | 70 |
| 0.31 | 29 | 0.2831 | 0.2672 | 9 | 0.3025 | 95 |
| 0.34 | 30 | 0.3176 | 0.2998 | 10 | 0.3350 | 125 |
| 0.37 | 31 | 0.3564 | 0.3365 | 11 | 0.3675 | 161 |
| none (0.40) | 32 | 0.4000 | 0.3776 | 12 | 0.4000 | 203 |

Each label is the smallest two-decimal value at or above the larger of the P/P_gm and B top edges, so every step adds at least one bin to every statistic. The labels aren't tight for each statistic separately, which keeps them consistent across statistics.

## v1 labels → equivalent v2 cut

| v1 label | P / P_gm bins | B k-bins / triangles | v2 cut with the same bins |
|---|---|---|---|
| 0.20 | 26 | 6 / 34 | 0.21 |
| 0.25 | 28 | 7 / 50 | P/P_gm 0.27; B 0.24 |
| 0.30 | 30 | 9 / 95 | P/P_gm 0.34; B 0.31 |
| 0.35 | 31 | 10 / 125 | P/P_gm 0.37; B 0.34 |
| 0.40 | 32 | 12 / 203 | none |

## Fiducial cuts (v2)

`_kp0.37 _kb0.24 _kpgm0.24`, set as `SWEEP_MASKS` and `FIDUCIAL_MASKS` in `generate_config_inference`.

- kp0.37 uses the same P bins as v1 kp0.35 (31 bins).
- kb0.24 uses the same B bins as v1 kb0.25 (7 bins, 50 triangles).
- kpgm0.24 has **one P_gm bin fewer** than v1 kpgm0.25: 27 bins instead of 28. v1 kpgm0.25 corresponds to v2 kpgm0.27.
- In the overall-k figure, all statistics share k up to 0.24, and only P(k) extends beyond that.

The noise A/B models (pk 31 + pgm 28 = 59 features) therefore correspond to v2 `_kp0.37_kpgm0.27`, not to the v2 fiducial `_kp0.37_kpgm0.24`.

Bin counts computed 2026-11-16 from `K_EDGES` and the k123 of a v2 bispec training file.
