"""plot_shame_bias_conventions.py

Cross-spectrum ratio P_{SHAMe x model} / P_model on the phase-matched mock (nbar = 2.2e-4),
for the deterministic muchisimocks tracer built with different conversions of the
probabilistic-bias (Stuecker+24) SHAMe values, and for the field-level fit.

  model m = delta_m + b1 delta + b2 delta^2 + bs2 s^2 + bl lapl(delta)   (no noise, no shot noise)
  ratio   = <t m*> / <m m*> per k-shell, t = SHAMe galaxy field; mean of phases 0 and pi.

Usage:  python plot_shame_bias_conventions.py [out.png]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

import utils_model
import compute_shame_matched_noisepoly as cp     # loaders, FFT P(k) estimator (dk = k_f), field-level fit

TAG = cp.TAG_MOCK                                 # '_nbar0.00022'
assert TAG == '_nbar0.00022'

# probabilistic-bias values as delivered (undo the /2 that utils_model.bias_dict_shame applies to b2, bs2)
d = utils_model.bias_dict_shame[TAG]
prob = dict(b1=d['b1'], b2=d['b2'] * 2, bs2=d['bs2'] * 2, bl=d['bl'])
fit = dict(zip(['b1', 'b2', 'bs2', 'bl'], cp.fit_biases()))     # phase mean, k <= 0.2

P = r'^\mathrm{prob}'
SETS = [
    (rf'$b_2 = 2\,b_2{P},\ b_{{s^2}} = 2\,b_{{s^2}}{P}$  (old code)',
     [prob['b1'], 2 * prob['b2'], 2 * prob['bs2'], prob['bl']], 'C3', '-'),
    (rf'$b_2 = b_2{P},\ b_{{s^2}} = b_{{s^2}}{P}$  (as delivered)',
     [prob['b1'], prob['b2'], prob['bs2'], prob['bl']], 'C1', '--'),
    (rf'$b_2 = b_2{P}/2,\ b_{{s^2}} = b_{{s^2}}{P}$  (Stücker+24 conv., $b_{{K^2}}$)',
     [prob['b1'], prob['b2'] / 2, prob['bs2'], prob['bl']], 'C0', '-'),
    (rf'$b_2 = b_2{P}/2,\ b_{{s^2}} = b_{{s^2}}{P}/2$  (Stücker+24 conv., $b_J$; code)',
     [prob['b1'], prob['b2'] / 2, prob['bs2'] / 2, prob['bl']], 'C2', '--'),
    (r'field-level fit to SHAMe ($k\leq0.2$), all four $b$',
     [fit['b1'], fit['b2'], fit['bs2'], fit['bl']], 'k', ':'),
]


def main(fn_out=None):
    # fn_out=None: don't save, just return the figure (for notebooks)
    ratios = {lab: [] for lab, *_ in SETS}
    for phase in cp.PHASES:
        ops_k = [cp.ft(o) for o in cp.load_ops(phase)]
        t_k = cp.ft(cp.load_shame_tracer(phase))
        for lab, b, _, _ in SETS:
            m_k = ops_k[0] + sum(bi * o for bi, o in zip(b, ops_k[1:]))
            ratios[lab].append(cp.pk(t_k, m_k) / cp.pk(m_k, m_k))
    k = cp.k_pk

    fig, ax = plt.subplots(figsize=(9, 5.2), layout='constrained')
    for lab, b, c, ls in SETS:
        ax.plot(k, np.mean(ratios[lab], axis=0), color=c, ls=ls, lw=1.6, label=lab)
    ax.axhline(1, color='0.5', lw=0.8)
    ax.set_xlim(0.01, 0.38); ax.set_ylim(0.8, 1.06)
    ax.set_xlabel(r'$k$ [$h$/Mpc]')
    ax.set_ylabel(r'$P_{\mathrm{SHAMe}\times\mathrm{model}}(k)\ /\ P_\mathrm{model}(k)$')
    ax.set_title(r'SHAMe $\bar n = 2.2\times10^{-4}$ vs phase-matched muchisimocks (deterministic tracer, no noise)', fontsize=10)
    ax.legend(fontsize=8.5, loc='lower left', title=r'$b_1,\ b_{\nabla^2} = b^\mathrm{prob}$ unless noted', title_fontsize=8.5)

    txt = ('                 $b_1$      $b_2$      $b_{s^2}$     $b_{\\nabla^2}$\n'
           f'prob. (as given)  {prob["b1"]:.3f}   {prob["b2"]:+.3f}   {prob["bs2"]:+.3f}   {prob["bl"]:+.3f}\n'
           f'field-level fit   {fit["b1"]:.3f}   {fit["b2"]:+.3f}   {fit["bs2"]:+.3f}   {fit["bl"]:+.3f}')
    ax.text(0.99, 0.98, txt, transform=ax.transAxes, ha='right', va='top', family='monospace', fontsize=8.5,
            bbox=dict(boxstyle='round', fc='white', ec='0.7'))
    if fn_out is not None:
        fig.savefig(fn_out, dpi=130)
        print(f'saved {fn_out}')

    kq = [0.1, 0.2, 0.3, 0.35]; iq = [np.argmin(np.abs(k - q)) for q in kq]
    print('prob (as delivered):', {n: round(v, 4) for n, v in prob.items()})
    print('field-level fit    :', {n: round(v, 4) for n, v in fit.items()})
    print('ratio at k =', kq)
    for lab, b, _, _ in SETS:
        print(f'  {np.round(np.mean(ratios[lab], axis=0)[iq], 3)}   b = {np.round(b, 4)}   {lab}')
    return fig


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    main(sys.argv[1] if len(sys.argv) > 1 else 'shame_bias_conventions.png')
