"""compute_shame_matched_noisebest.py

Populate the phase-matched muchisimocks fields (make_shame_matched_mock.py) the same
way the fixed-SHAMe-cosmology cosmic-variance test set is populated in the paper's
"noisy" scenario (generate_config_inference._TEST_SCENARIO_TAGS):
  tag_biasparams = '_biasshame{nbar}_noisebest_p0_n1'   (SHAMe biases for that density + best-fit
                                                         multiplicative noise amplitudes)
  noise fields   = unit Gaussian, multiplicative model (utils_model.get_tracer_field)
then compute P, P_gm, B (and P, P_gm Gaussian errors) with the production estimators, for
N_REAL noise realizations per phase. make_shame_matched_testset.py turns the output into a
muchisimocks-layout test set (_shamematched{nbar}_p0_n10).

Differences from the library pipeline, forced by the matched 1024 / 132^3 grid:
  - unit noise fields are drawn at 132^3 (fresh seeds, SEED_NOISE_BASE + i)
  - A_noise is rescaled by (V_cell,lib / V_cell,matched)^(1/2) = 1.0107 so the physical
    noise power (A^2 V_cell for white noise) matches the library's 1000/128 cells

Output: data/data_shame/matched_seed-406168/stats_noisebest{nbar}.npy (--nbar, default the fiducial
        0.00022; truth _biasshame{nbar}_noisebest_p0_n1 = bias_dict_shame['_nbar{nbar}'] + that density's
        fixed-b noise fit, written by select_shame_noisebest.py write)
Run on a compute node:  python compute_shame_matched_noisebest.py [--nbar 0.00011]
  --no-rescale : use the library A_noise unchanged (all 5 amplitudes), saved to
                 stats_noisebest_norescale.npy (check of the V_cell rescaling)
  --noise-m2   : second-order multiplicative noise instead (_biasshame_noisem2best_p0_n1,
                 An_homog eps O_1 + An2_homog eps^2 O_1 + An2_bl eps^2 O_lapl), A not rescaled
                 (the eps^2 terms are partly a mean shift, not white noise); saved to
                 stats_noisem2best.npy
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

import paths
import utils_model
import data_loader
import compute_statistics as cs
import make_shame_matched_mock as msm

SEED = -406168
L, NG, NG_DEP = msm.BOX_SIZE, msm.N_GRID_TARGET, msm.N_GRID
NOISE_M2 = '--noise-m2' in sys.argv
assert not ('--nbar' in sys.argv and NOISE_M2), '--nbar is for the linear multiplicative noise model only'
NBAR = sys.argv[sys.argv.index('--nbar') + 1] if '--nbar' in sys.argv else '0.00022'  # fiducial density
TAG_BIASPARAMS = ('_biasshame_noisem2best_p0_n1' if NOISE_M2 else
                  f'_biasshame{NBAR}_noisebest_p0_n1')
NOISE_MODEL = 'multiplicative_m2p3' if NOISE_M2 else 'multiplicative'
NOISE_NAMES = utils_model.noiseparam_names_ordered_m2p3 if NOISE_M2 else utils_model.noiseparam_names_ordered
TAG_MOCK = f'_nbar{NBAR}'   # the density the bias + noise best fit was made for
PHASES = ['0', 'pi']
N_REAL = 10
SEED_NOISE_BASE = 40000
N_THREADS = 16
NO_RESCALE = ('--no-rescale' in sys.argv) or NOISE_M2
V_SCALE = 1.0 if NO_RESCALE else ((1000. / 128) / (L / NG))**1.5
FN_OUT = msm.DIR_DATA_SHAME / f'matched_seed{SEED}' / (
    'stats_noisem2best.npy' if NOISE_M2 else
    'stats_noisebest_norescale.npy' if NO_RESCALE else
    f'stats_noisebest{NBAR}.npy')


def main():
    cosmo = utils_model.get_cosmo(utils_model.cosmo_dict_shame)
    _, pdict = data_loader.load_bias_params(TAG_BIASPARAMS)
    b = [pdict[n] for n in utils_model.biasparam_names_ordered]
    if NBAR:   # the truth file must carry this density's fiducial SHAMe biases
        assert np.allclose(b, [utils_model.bias_dict_shame[TAG_MOCK][n] for n in utils_model.biasparam_names_ordered]), b
    A = [pdict[n] * V_SCALE for n in NOISE_NAMES]
    print(f'{TAG_BIASPARAMS}: b={b}, A_noise (rescaled x{V_SCALE:.4f})={A}', flush=True)

    base = cs.setup_bispec(L, NG, n_threads=N_THREADS)
    import PolyBin3D as pb
    k_edges = np.linspace(0.01, 0.4, 13)
    bspec = pb.BSpec(base, k_edges, applySinv=None, mask=None, lmax=1,
                     k_bins_squeeze=k_edges.copy(), include_partial_triangles=False)
    out = dict(k123=bspec.get_ks(), tag_biasparams=TAG_BIASPARAMS, tag_mock=TAG_MOCK,
               b=b, A_noise=A, noise_names=list(NOISE_NAMES), noise_model=NOISE_MODEL,
               V_scale=V_SCALE, n_real=N_REAL)

    for phase in PHASES:
        fn = paths.mocks_lib_ood_dir(msm.SUBDIR_OOD) / f'phase{phase}_seed{SEED}' / f'bias_fields_eul_deconvolved_{SEED}.npy'
        fields = np.load(fn)
        dens = np.load(msm.DIR_DATA_SHAME / f'dens_mesh_phase{phase}.npy')
        # P_gm against three "matter" fields:
        #   nbody    : Rings N-body matter mesh (what the SHAMe P_gm uses)
        #   m2m      : map2map nonlinear matter = the '1' operator, fields[0]
        #   delta_op : the advected linear delta operator, fields[1] (what compute_statistics.py uses)
        matter = {'nbody': (dens / np.sum(dens)).astype(np.float32),
                  'm2m': (fields[0] / NG_DEP**3).astype(np.float32),
                  'delta_op': (fields[1] / NG_DEP**3).astype(np.float32)}
        res = dict(pk=[], pk_err=[], pgm={m: [] for m in matter}, pgm_err={m: [] for m in matter}, bk=[])
        for i in range(N_REAL):
            t0 = time.time()
            # same seed for a given i in both phases, as in the paired library
            eps = np.random.default_rng(SEED_NOISE_BASE + i).standard_normal(fields[0].shape)
            tf = utils_model.get_tracer_field(fields, b, NG_DEP, noise_field=eps, A_noise=A,
                                              noise_model=NOISE_MODEL).astype(np.float32)
            pk = cs.compute_pk(tf, cosmo, L, n_threads=N_THREADS)
            res['k'] = pk['k']
            res['pk'].append(pk['pk'])
            res['pk_err'].append(pk['pk_gaussian_error'])
            for m in matter:
                pgm = cs.compute_pgm(tf, matter[m], cosmo, L, n_threads=N_THREADS)
                res['pgm'][m].append(pgm['pk'])
                res['pgm_err'][m].append(pgm['pk_gaussian_error'])
            res['bk'].append(bspec.Bk_ideal(tf, discreteness_correction=True)['b0'])
            print(f'phase={phase} real={i}: {time.time() - t0:.0f} s', flush=True)
        out[phase] = {k: (np.array(v) if not isinstance(v, dict) else {m: np.array(x) for m, x in v.items()})
                      for k, v in res.items()}
        np.save(FN_OUT, out)
    print(f'Saved {FN_OUT}')


if __name__ == '__main__':
    main()
