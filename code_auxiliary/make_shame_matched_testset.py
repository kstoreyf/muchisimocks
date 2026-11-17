"""make_shame_matched_testset.py

Write the phase-matched muchisimocks mock (SHAMe Rings ICs, populated like the fixed-SHAMe-cosmology
test set: bias_dict_shame + noisebest, 10 noise realizations) as an ordinary muchisimocks test set,
so run_inference.py's in-distribution test path loads it (individual realizations, or their mean
with evaluate_mean: true).

  tag_params_test     = _shamematched_p0_n10          (SHAMe cosmology, fixed)
  tag_biasparams_test = _biasshame_noisebest_p0_n1    (truth: SHAMe biases + noisebest, library units)
  tag_noise_test      = _noisematched                 (eps seeds 40000 + i, same for both phases)

Observation i = pair mean over the 0/pi phases of noise realization i, as the SHAMe test uses the
pair mean. P_gm is against the '1' operator (map2map matter), as in v2 training. B is computed on the
132^3 matched grid; the loader normalizes muchisimocks B with n_grid = 128 (load_bispec), so b0 is
stored times (132/128)^9 to give the same physical B.

Input:  data/data_shame/matched_seed-406168/stats_noisebest.npy (compute_shame_matched_noisebest.py)
Output: $MUCHISIMOCKS_DATA/{pks,pgms,bispecs}_mlib/..._shamematched_p0_n10_biasshame_noisebest_p0_n1_noisematched/
        data/params/params_fixed_shamematched_p0_n10.txt
With --nbar <nbar> (other SHAMe densities): tag_params_test = _shamematched{nbar}_p0_n10,
tag_biasparams_test = _biasshame{nbar}_noisebest_p0_n1, input stats_noisebest{nbar}.npy
(compute_shame_matched_noisebest.py --nbar <nbar>).
Run with code_auxiliary/env_v2.sh sourced (seconds; login node is fine):  python make_shame_matched_testset.py [--nbar 0.00011]
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

import data_loader
import make_shame_matched_mock as msm

SEED = -406168
NBAR = sys.argv[sys.argv.index('--nbar') + 1] if '--nbar' in sys.argv else None
TAG_PARAMS = f'_shamematched{NBAR}_p0_n10' if NBAR else '_shamematched_p0_n10'
TAG_BIASPARAMS = f'_biasshame{NBAR}_noisebest_p0_n1' if NBAR else '_biasshame_noisebest_p0_n1'
TAG_NOISE = '_noisematched'
N_GRID_LOADER = 128            # data_loader.load_bispec(..., n_grid=128) for muchisimocks
FN_IN = msm.DIR_DATA_SHAME / f'matched_seed{SEED}' / (f'stats_noisebest{NBAR}.npy' if NBAR else 'stats_noisebest.npy')
DIR_PARAMS = HERE.parent / 'data' / 'params'


def main():
    assert os.environ.get('MUCHISIMOCKS_DATA', '').endswith('data_v2'), 'source code_auxiliary/env_v2.sh first'
    st = np.load(FN_IN, allow_pickle=True).item()
    assert st['tag_biasparams'] == TAG_BIASPARAMS, st['tag_biasparams']
    n_real = st['n_real']
    assert n_real == int(TAG_PARAMS.split('_n')[-1]), (n_real, TAG_PARAMS)
    ph = [st['0'], st['pi']]
    pair = lambda f: 0.5 * (f(ph[0]) + f(ph[1]))       # (n_real, nbins)
    b_scale = (msm.N_GRID_TARGET / N_GRID_LOADER)**9

    stats = {
        'pk': dict(k=ph[0]['k'], y=pair(lambda d: d['pk']), err=pair(lambda d: d['pk_err'])),
        'pgm': dict(k=ph[0]['k'], y=pair(lambda d: d['pgm']['m2m']), err=pair(lambda d: d['pgm_err']['m2m'])),
    }
    k123 = st['k123']
    bk = pair(lambda d: d['bk']) * b_scale

    for statistic in ('pk', 'pgm', 'bispec'):
        d = data_loader.get_dir_statistics(statistic, TAG_PARAMS, TAG_BIASPARAMS, tag_noise=TAG_NOISE)
        Path(d).mkdir(parents=True, exist_ok=True)
        for i in range(n_real):
            if statistic == 'bispec':
                obj = dict(k123=k123, bispectrum={'b0': bk[i]}, weight=k123.prod(axis=0))
            else:
                s = stats[statistic]
                obj = dict(k=s['k'], pk=s['y'][i], pk_gaussian_error=s['err'][i])
            np.save(f'{d}/{statistic}_{i}.npy', obj)
        print(f'wrote {n_real} {statistic} files to {d}')

    fn_cosmo = DIR_PARAMS / f'params_fixed{TAG_PARAMS}.txt'
    shutil.copy(DIR_PARAMS / 'params_fixed_shame_p0_n1000.txt', fn_cosmo)   # SHAMe cosmology
    print(f'wrote {fn_cosmo}')


if __name__ == '__main__':
    main()
