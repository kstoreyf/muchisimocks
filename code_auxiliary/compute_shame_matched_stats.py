"""compute_shame_matched_stats.py

Compute P(k), P_gm(k) and B(k1,k2,k3) for the phase-matched muchisimocks tracer
fields (see make_shame_matched_mock.py), with the same production functions and
binning used for the SHAMe data vectors (process_mock_catalogs.py):
cs.compute_pk / cs.compute_pgm / PolyBin3D Bk_ideal, box 1024, 132^3 grid.

Model variants, per nbar and phase:
  dict        fiducial utils_model.bias_dict_shame biases, no noise
  fit         biases fit at field level to the SHAMe tracer (k <= KMAX_FIT), no noise
  dict_noise  dict + multiplicative white noise (1+delta_m) * A * eps,
  fit_noise   fit  + same noise;  A set so P_noise = 1/nbar (Poisson level)
P_gm is computed against three "matter" fields:
  nbody  the SHAMe N-body matter mesh (what the SHAMe data vector uses)
  O0     map2map nonlinear matter (bias operator "1")
  O1     the delta operator, bias_terms_eul[1] (what compute_statistics.py uses
         for muchisimocks P_gm in production)

Output: data/data_shame/matched_seed-406168/stats.npy (dict)

Run on a compute node (bacco/PolyBin3D), ~1 h on 16 threads:
  python compute_shame_matched_stats.py
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
import compute_statistics as cs
import make_shame_matched_mock as msm

SEED = -406168
L = msm.BOX_SIZE            # 1024
NG = msm.N_GRID_TARGET      # 132
NG_DEP = msm.N_GRID         # 524, deposit grid -> normalization as get_tracer_field(n_grid_norm=524)
TAGS = ['_nbar0.00011', '_nbar0.00022', '_nbar0.00054']
PHASES = ['0', 'pi']
KMAX_FIT = 0.2
N_THREADS = 16
SEED_NOISE = 12345
FN_OUT = msm.DIR_DATA_SHAME / f'matched_seed{SEED}' / 'stats.npy'


def kmag_grid():
    kf = 2 * np.pi / L
    kx = np.fft.fftfreq(NG, d=1. / NG) * kf
    kz = np.fft.rfftfreq(NG, d=1. / NG) * kf
    return np.sqrt(kx[:, None, None]**2 + kx[None, :, None]**2 + kz[None, None, :]**2)


def fit_bias_fieldlevel(ops, tracer, kmax, kmag):
    """Least-squares b (b1, b2, bs2, bl) minimizing |t - O0 - sum b_i O_i|^2 over modes k<=kmax."""
    ops_k = [np.fft.rfftn(o) for o in ops]
    tk = np.fft.rfftn(tracer)
    m = (kmag <= kmax) & (kmag > 0)
    y = (tk - ops_k[0])[m]
    A = np.stack([o[m] for o in ops_k[1:]], axis=1)
    b, *_ = np.linalg.lstsq(np.concatenate([A.real, A.imag]), np.concatenate([y.real, y.imag]), rcond=None)
    return b


def main():
    cosmo = utils_model.get_cosmo(utils_model.cosmo_dict_shame)
    kmag = kmag_grid()
    base = cs.setup_bispec(L, NG, n_threads=N_THREADS)
    import PolyBin3D as pb
    k_edges = np.linspace(0.01, 0.4, 13)
    bspec = pb.BSpec(base, k_edges, applySinv=None, mask=None, lmax=1,
                     k_bins_squeeze=k_edges.copy(), include_partial_triangles=False)
    k123 = bspec.get_ks()
    out = dict(k123=k123, weight=k123.prod(axis=0), seed=SEED, kmax_fit=KMAX_FIT,
               bias_names=list(utils_model.biasparam_names_ordered))

    def bk(field):
        return bspec.Bk_ideal(field, discreteness_correction=True)['b0']

    rng = np.random.default_rng(SEED_NOISE)
    for phase in PHASES:
        fn = paths.mocks_lib_ood_dir(msm.SUBDIR_OOD) / f'phase{phase}_seed{SEED}' / f'bias_fields_eul_deconvolved_{SEED}.npy'
        fields = np.load(fn)                               # (5, 132,132,132), real space
        ops_norm = fields.astype(np.float64) / NG_DEP**3   # same units as SHAMe tracer_field (delta/132^3)
        dens = np.load(msm.DIR_DATA_SHAME / f'dens_mesh_phase{phase}.npy')
        matter = {'nbody': dens / np.sum(dens),            # as process_mock_catalogs.compute_pgm
                  'O0': ops_norm[0], 'O1': ops_norm[1]}    # O1 as compute_statistics.py pgm
        onepdelta = fields[0] / fields[0].mean()
        eps = rng.standard_normal(fields[0].shape)         # one noise realization per phase, shared by nbars
        v_cell = (L / NG)**3
        matter = {m: v.astype(np.float32) for m, v in matter.items()}
        out[('matter_pk', phase)] = {m: cs.compute_pk(matter[m], cosmo, L, n_threads=N_THREADS)['pk'] for m in matter}
        for tag in TAGS:
            t0 = time.time()
            nbar = float(tag.split('nbar')[1])
            tracer = np.load(msm.DIR_DATA_SHAME / f'data{tag}' / f'tracer_field_phase{phase}.npy')
            b_dict = np.array([utils_model.bias_dict_shame[tag][n] for n in utils_model.biasparam_names_ordered])
            b_fit = fit_bias_fieldlevel(ops_norm, tracer, KMAX_FIT, kmag)
            amp = 1. / np.sqrt(nbar * v_cell * np.mean(onepdelta**2))
            noise = onepdelta * amp * eps / NG**3          # (1+delta_m) A eps, in delta/132^3 units
            res = dict(nbar=nbar, b_dict=b_dict, b_fit=b_fit, A_noise=amp)
            for lab, b in [('dict', b_dict), ('fit', b_fit)]:
                det = utils_model.get_tracer_field(fields, b, NG_DEP)
                for suffix, tf in [('', det), ('_noise', det + noise)]:
                    tf = tf.astype(np.float32)
                    pk = cs.compute_pk(tf, cosmo, L, n_threads=N_THREADS)
                    res[lab + suffix] = dict(
                        pk=pk['pk'],
                        pgm={m: cs.compute_pgm(tf, matter[m], cosmo, L, n_threads=N_THREADS)['pk'] for m in matter},
                        bk=bk(tf))
                    res['k'] = pk['k']
            # SHAMe tracer x {O0, O1} for reference (stored SHAMe pgm uses nbody)
            res['shame_pgm'] = {m: cs.compute_pgm(tracer, matter[m], cosmo, L, n_threads=N_THREADS)['pk'] for m in matter}
            out[(tag, phase)] = res
            print(f'phase={phase} {tag}: b_fit={np.round(b_fit, 3)}, A_noise={amp:.3f}, {time.time() - t0:.0f} s', flush=True)
            FN_OUT.parent.mkdir(parents=True, exist_ok=True)
            np.save(FN_OUT, out)
    print(f'Saved {FN_OUT}')


if __name__ == '__main__':
    main()
