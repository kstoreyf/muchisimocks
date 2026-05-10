"""make_quijote_matched_mocks.py

Generate Quijote-matched muchisimocks bias fields and compute their statistics
(P(k), P_gm(k), B(k)) for direct comparison against the matching Quijote ground
truth. Mirrors code/data_creation_pipeline.py and code/compute_statistics.py
but operates on Yin's existing Quijote LH directories at
/scratch/kstoreyf/Yin_data/Quijote.

For each LH:

  Stage 1 (bias fields, --do_bias_fields, default on):
    - Reads Yin's `dis_<idx>.npy` (Quijote ground truth) and
      `pred_pos_<idx>.npy` (muchisimocks/map2map prediction made from the same
      Quijote ICs).
    - Runs the muchisimocks pipeline (BiasModel + k-cut + deconvolution) on each.
    - Saves OUR outputs alongside Yin's data as
      `bias_fields_eul_<src>_deconvolved<damping_tag>_<idx:04d>.npy` where
      src in {quijote, muchisimocks}.
    - Never overwrites Yin's existing files (filename starts with bias_fields_eul_).

  Stage 2 (statistics, --do_statistics, default on):
    - Loads the Stage-1 bias fields plus the Quijote LH cosmology and the SHAMe
      OOD bias values (via data_loader.load_params_ood).
    - Computes pk, pgm, bispec via compute_statistics.compute_*.
    - Saves to /scratch/kstoreyf/muchisimocks/data/<stat>s_mlib/<stat>s_quijote_matched/
      <stat>_<idx:04d>_<src>_b<tag_bias><damping_tag>.npy
      where tag_bias = f'shame{tag_mock_shame}'.

Both stages can be toggled independently.

`--include_damping` toggles between two BiasModel call patterns (these produce
different bias fields, see notebook discussion):
  - default (off): pass `linear_delta=dens_lin` to BiasModel. The
    `damping_scale=0.75` argument is then silently ignored by bacco
    (you'll see a warning). Matches notebooks/2024-04-03_check_map2map.ipynb.
  - on: build a stub bacco LPT sim, replace its `linear_field[0]` with Yin's
    Quijote `dens_lin`, and pass `sim=...` (no `linear_delta`). The damping
    scale is then applied. Matches the BiasModel call in
    code/data_creation_pipeline.py:predicted_positions_to_bias_fields.
  - When on, all output filenames get the suffix `_damping<scale>` so both
    versions can coexist on disk.

Examples:
  # All LHs available in Yin's Quijote dir, both stages (default)
  python make_quijote_matched_mocks.py

  # Just LH 663 and 37
  python make_quijote_matched_mocks.py --idxs_LH 663 37

  # Skip bias-field gen, just (re)compute stats
  python make_quijote_matched_mocks.py --no_bias_fields

  # Apply the damping scale (matches code/data_creation_pipeline.py)
  python make_quijote_matched_mocks.py --include_damping

  # Current LHs in scratch: 0037 0574 0822 1082 1510 0254 0663 0977 1317 1642
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import bacco

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1] if HERE.parent.name in ("code", "code_auxiliary") else HERE.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

import compute_statistics
import data_creation_pipeline
import data_loader
import paths
import utils_model


# ----------------------------------------------------------------------
# Defaults / constants
# ----------------------------------------------------------------------
DEFAULT_TAG_MOCKS = "_quijote_matched"
DEFAULT_TAG_MOCK_SHAME = "_nbar0.00022"
DEFAULT_STATS = ["pk", "pgm", "bispec"]

DIR_QUIJOTE_BASE = Path("/scratch/kstoreyf/Yin_data/Quijote")

# Match the muchisimocks library grid settings.
BOX_SIZE = 1000.0
N_GRID = 512
N_GRID_TARGET = 128
DAMPING_SCALE = 0.75

SOURCES = ("quijote", "muchisimocks")

# Quijote LH param file order: [omega_m, omega_baryon, h, ns, sigma8]
# (see notebooks/2024-04-03_check_map2map.ipynb)
QUIJOTE_PARAM_NAMES = ["omega_m", "omega_baryon", "hubble", "ns", "sigma8_cold"]


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
def _yin_lh_dir(idx_LH):
    return DIR_QUIJOTE_BASE / f"LH{idx_LH:04d}"


def _disp_path(idx_LH, source):
    """Yin's existing displacement file for the given source."""
    name = {"quijote": "dis", "muchisimocks": "pred_pos"}[source]
    return _yin_lh_dir(idx_LH) / f"{name}_{idx_LH:04d}.npy"


def _lin_den_path(idx_LH):
    return _yin_lh_dir(idx_LH) / f"lin_den_{idx_LH:04d}.npy"


def _params_path(idx_LH):
    return _yin_lh_dir(idx_LH) / f"param_{idx_LH:04d}.txt"


def _damping_tag(include_damping, damping_scale=DAMPING_SCALE):
    """Filename suffix for bias fields / stats computed with the damping codepath."""
    return f"_damping{damping_scale}" if include_damping else ""


def _bfields_path(idx_LH, source, include_damping=False):
    """OUR output: bias fields written into Yin's Quijote LH dir."""
    tag_d = _damping_tag(include_damping)
    return _yin_lh_dir(idx_LH) / f"bias_fields_eul_{source}_deconvolved{tag_d}_{idx_LH:04d}.npy"


def _stat_path(statistic, idx_LH, source, tag_bias, tag_mocks, include_damping=False):
    tag_d = _damping_tag(include_damping)
    dir_stat = paths.statistics_dir(statistic, tag_mocks)
    dir_stat.mkdir(parents=True, exist_ok=True)
    return dir_stat / f"{statistic}_{idx_LH:04d}_{source}_b{tag_bias}{tag_d}.npy"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def find_available_lhs():
    return sorted(int(d.name[2:]) for d in DIR_QUIJOTE_BASE.glob("LH*") if d.is_dir())


def _load_quijote_cosmo_params(idx_LH):
    vals = np.loadtxt(_params_path(idx_LH))
    return dict(zip(QUIJOTE_PARAM_NAMES, vals))


def _load_shame_bias(tag_mock_shame):
    """Returns (bias_vector, tag_bias_str). Reads SHAMe OOD bias dict via
    data_loader.load_params_ood (which sources from utils_model.bias_dict_shame)."""
    shame_param_dict = data_loader.load_params_ood("shame", tag_mock_shame)
    bias_vector = [shame_param_dict[n] for n in utils_model.biasparam_names_ordered]
    tag_bias = f"shame{tag_mock_shame}"
    return bias_vector, tag_bias


# ----------------------------------------------------------------------
# Stage 1: bias fields
# ----------------------------------------------------------------------
def displacements_to_deconvolved_bfields(disp_field, dens_lin,
                                         n_grid=N_GRID, n_grid_target=N_GRID_TARGET,
                                         box_size=BOX_SIZE, damping_scale=DAMPING_SCALE,
                                         include_damping=False, cosmo=None, seed=0):
    """Stripped-down version of `predicted_positions_to_bias_fields` from
    code/data_creation_pipeline.py.

    Differences from the original:
    - No I/O (no `dir_LH`, no intermediate save)
    - Reuses `data_creation_pipeline.deconvolve_bias_field` and
      `utils_model.remove_highk_modes`.

    Two BiasModel call patterns, controlled by `include_damping`:
    - False (default): pass `linear_delta=dens_lin` directly. Mirrors
      notebooks/2024-04-03_check_map2map.ipynb. The `damping_scale` argument is
      then silently ignored by bacco.
    - True: build a stub bacco LPT sim from `cosmo`+`seed`, replace its
      `linear_field[0]` with the externally provided `dens_lin`, and pass
      `sim=...` (no `linear_delta`). The `damping_scale` is then applied.
      Mirrors the BiasModel call in
      data_creation_pipeline.predicted_positions_to_bias_fields.
      Requires `cosmo` to be passed.
    """
    interlacing = False

    grid = bacco.visualization.uniform_grid(npix=n_grid, L=box_size, ndim=3, bounds=False)
    pred_pos = bacco.scaler.add_displacement(
        None, disp_field, box=box_size, pos=grid.reshape(-1, 3),
        vel=None, vel_factor=0, verbose=False,
    )[0]

    if include_damping:
        # Build a stub LPT sim for the BiasModel `sim=` codepath, then overwrite
        # its linear field with Yin's externally provided `dens_lin`. We don't
        # use the disp returned by create_lpt_simulation; we already have
        # `pred_pos` from the externally provided displacement field.
        # Mirrors `bacco.utils.create_lpt_simulation` arguments in
        # data_creation_pipeline.run_single.
        assert cosmo is not None, "cosmo is required when include_damping=True"
        # Without these, create_lpt_simulation builds the linear field at the
        # default scaling.disp_ngrid (128), not at Nmesh. Same pattern as in
        # data_creation_pipeline.run_single.
        bacco.configuration.update({"pknbody": {"ngrid": n_grid}})
        bacco.configuration.update({"scaling": {"disp_ngrid": n_grid}})
        sim, _ = bacco.utils.create_lpt_simulation(
            cosmo, box_size, Nmesh=n_grid, Seed=seed,
            FixedInitialAmplitude=False, InitialPhase=0,
            expfactor=1.0, LPT_order=1, order_by_order=None,
            phase_type=1, ngenic_phases=True, return_disp=True,
            sphere_mode=0,
        )
        # NOTE: assumes Yin's dens_lin is in the same units as sim.linear_field[0]
        # (i.e. what BiasModel expects for `linear_delta`). If results look wrong,
        # this normalization is the first thing to check.
        sim.linear_field[0] = dens_lin.astype(sim.linear_field[0].dtype)

        bmodel = bacco.BiasModel(
            sim=sim,
            ngrid=n_grid, ngrid1=None,
            BoxSize=box_size, sdm=False, mode="dm",
            npart_for_fake_sim=n_grid, damping_scale=damping_scale,
            bias_model="expansion", deposit_method="cic",
            use_displacement_of_nn=False, interlacing=interlacing,
        )
    else:
        bmodel = bacco.BiasModel(
            sim=None, linear_delta=dens_lin,
            ngrid=n_grid, ngrid1=None,
            BoxSize=box_size, sdm=False, mode="dm",
            npart_for_fake_sim=n_grid, damping_scale=damping_scale,
            bias_model="expansion", deposit_method="cic",
            use_displacement_of_nn=False, interlacing=interlacing,
        )
    bias_fields_lag = bmodel.bias_terms_lag()

    bias_terms_eul = []
    for ii in range(len(bias_fields_lag)):
        bt = bacco.statistics.compute_mesh(
            ngrid=n_grid, box=box_size, pos=pred_pos,
            mass=bias_fields_lag[ii].flatten(),
            deposit_method="cic", interlacing=interlacing,
        )
        bias_terms_eul.append(bt)
    bias_terms_eul = np.squeeze(np.array(bias_terms_eul))

    bias_terms_kcut = np.array([
        utils_model.remove_highk_modes(bt, box_size, n_grid_target)
        for bt in bias_terms_eul
    ])

    bias_terms_kcut_deconv = data_creation_pipeline.deconvolve_bias_field(
        bias_terms_kcut, n_grid
    ).astype(np.float32)
    return bias_terms_kcut_deconv


def create_bias_fields_for_lh(idx_LH, sources, include_damping=False, overwrite=False):
    print(f"\n[bias_fields] LH{idx_LH:04d}")
    fn_lin = _lin_den_path(idx_LH)
    if not fn_lin.exists():
        print(f"  missing lin_den ({fn_lin}); skip LH")
        return

    cosmo = None
    if include_damping:
        fn_params = _params_path(idx_LH)
        if not fn_params.exists():
            print(f"  missing param file ({fn_params}); skip LH (needed for include_damping)")
            return
        cosmo = utils_model.get_cosmo(_load_quijote_cosmo_params(idx_LH),
                                      a_scale=1.0, sim_name="quijote")

    dens_lin = None
    for source in sources:
        fn_out = _bfields_path(idx_LH, source, include_damping=include_damping)
        # Safety: never overwrite Yin's existing files (we control the name, but assert).
        assert fn_out.name.startswith("bias_fields_eul_"), f"Unexpected output name: {fn_out}"
        if fn_out.exists() and not overwrite:
            print(f"  [{source}] {fn_out.name} exists; skip.")
            continue
        fn_disp = _disp_path(idx_LH, source)
        if not fn_disp.exists():
            print(f"  [{source}] missing input {fn_disp.name}; skip.")
            continue
        if dens_lin is None:
            dens_lin = np.load(fn_lin)[0]
        print(f"  [{source}] {fn_disp.name} -> {fn_out.name}")
        t0 = time.time()
        disp = np.load(fn_disp)
        bfields = displacements_to_deconvolved_bfields(
            disp, dens_lin, include_damping=include_damping, cosmo=cosmo, seed=idx_LH,
        )
        del disp
        fn_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(fn_out, bfields)
        print(f"  [{source}] saved shape {bfields.shape} in {time.time() - t0:.1f}s")


# ----------------------------------------------------------------------
# Stage 2: statistics
# ----------------------------------------------------------------------
def compute_stats_for_lh(idx_LH, sources, statistics, bias_vector, tag_bias,
                         tag_mocks, base_bispec, n_threads,
                         include_damping=False, overwrite=False):
    print(f"\n[stats] LH{idx_LH:04d}")
    fn_params = _params_path(idx_LH)
    if not fn_params.exists():
        print(f"  missing param file ({fn_params}); skip LH")
        return

    cosmo = utils_model.get_cosmo(_load_quijote_cosmo_params(idx_LH),
                                  a_scale=1.0, sim_name="quijote")

    for source in sources:
        fn_bf = _bfields_path(idx_LH, source, include_damping=include_damping)
        if not fn_bf.exists():
            print(f"  [{source}] missing bfields ({fn_bf.name}); skip source")
            continue

        # Skip the (expensive) tracer/matter build if everything for this source
        # is already on disk.
        needed = [
            s for s in statistics
            if overwrite or not _stat_path(
                s, idx_LH, source, tag_bias, tag_mocks, include_damping=include_damping
            ).exists()
        ]
        if not needed:
            print(f"  [{source}] all stats exist; skip.")
            continue

        bfields = np.load(fn_bf)
        # Same conventions as compute_statistics.make_tracer_field / 'pgm' branch.
        tracer = utils_model.get_tracer_field(bfields, bias_vector, N_GRID)
        matter = bfields[1] / N_GRID ** 3

        for stat in statistics:
            fn_stat = _stat_path(
                stat, idx_LH, source, tag_bias, tag_mocks, include_damping=include_damping,
            )
            if fn_stat.exists() and not overwrite:
                print(f"  [{source}/{stat}] {fn_stat.name} exists; skip.")
                continue
            print(f"  [{source}/{stat}] -> {fn_stat.name}")
            t0 = time.time()
            if stat == "pk":
                compute_statistics.compute_pk(
                    tracer, cosmo, BOX_SIZE,
                    n_threads=n_threads, fn_stat=str(fn_stat))
            elif stat == "pgm":
                compute_statistics.compute_pgm(
                    tracer, matter, cosmo, BOX_SIZE,
                    n_threads=n_threads, fn_stat=str(fn_stat))
            elif stat == "bispec":
                assert base_bispec is not None, "base_bispec must be set up first"
                bspec, bk_corr = compute_statistics.compute_bispectrum(
                    base_bispec, tracer, fn_stat=None)
                # Use save_bispectrum directly so we record n_grid
                # (data_loader.load_bispec uses it for the norm**3 normalization).
                compute_statistics.save_bispectrum(
                    str(fn_stat), bspec, bk_corr, n_grid=N_GRID_TARGET)
            else:
                raise ValueError(f"Unknown statistic {stat!r}")
            print(f"  [{source}/{stat}] done in {time.time() - t0:.1f}s")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--idxs_LH", nargs="+", type=int, default=None,
                   help="LH indices to process. Default: all LH dirs found in DIR_QUIJOTE_BASE.")
    p.add_argument("--sources", nargs="+", default=list(SOURCES), choices=SOURCES,
                   help="Which mock source(s) to process.")
    p.add_argument("--statistics", nargs="+", default=DEFAULT_STATS,
                   choices=["pk", "pgm", "bispec"])
    p.add_argument("--tag_mock_shame", default=DEFAULT_TAG_MOCK_SHAME,
                   help="SHAMe OOD mock tag used for bias values (e.g. _nbar0.00022).")
    p.add_argument("--tag_mocks", default=DEFAULT_TAG_MOCKS,
                   help="Tag for the statistics output dir (e.g. _quijote_matched).")
    p.add_argument("--n_threads", type=int, default=4)
    p.add_argument("--no_bias_fields", dest="do_bias_fields", action="store_false",
                   help="Skip bias-field creation stage.")
    p.add_argument("--no_statistics", dest="do_statistics", action="store_false",
                   help="Skip statistics computation stage.")
    p.add_argument("--include_damping", action="store_true",
                   help="Apply BiasModel's damping_scale=%g (matches "
                        "code/data_creation_pipeline.py). Adds '_damping%g' tag "
                        "to all output filenames so it can coexist with the "
                        "no-damping outputs." % (DAMPING_SCALE, DAMPING_SCALE))
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files.")
    args = p.parse_args()
    if args.idxs_LH is None:
        args.idxs_LH = find_available_lhs()
        print(f"No --idxs_LH given; processing all LHs in {DIR_QUIJOTE_BASE}")
    print(f"idxs_LH:      {args.idxs_LH}")
    print(f"sources:      {args.sources}")
    print(f"statistics:   {args.statistics}")
    print(f"do_bias:      {args.do_bias_fields}")
    print(f"do_stats:     {args.do_statistics}")
    print(f"include_damping: {args.include_damping} (tag: {_damping_tag(args.include_damping)!r})")
    return args


def main():
    args = parse_args()

    if args.do_bias_fields:
        for idx_LH in args.idxs_LH:
            create_bias_fields_for_lh(
                idx_LH, args.sources,
                include_damping=args.include_damping, overwrite=args.overwrite,
            )

    if args.do_statistics:
        bias_vector, tag_bias = _load_shame_bias(args.tag_mock_shame)
        print(f"\nSHAMe bias vector ({utils_model.biasparam_names_ordered}): {bias_vector}")
        print(f"tag_bias: {tag_bias}")

        base_bispec = None
        if "bispec" in args.statistics:
            base_bispec = compute_statistics.setup_bispec(
                BOX_SIZE, N_GRID_TARGET, args.n_threads
            )

        for idx_LH in args.idxs_LH:
            compute_stats_for_lh(
                idx_LH, args.sources, args.statistics,
                bias_vector, tag_bias, args.tag_mocks,
                base_bispec, args.n_threads,
                include_damping=args.include_damping, overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
