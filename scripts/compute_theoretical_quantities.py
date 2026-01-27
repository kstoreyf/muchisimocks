import glob
import numpy as np
import pandas as pd
from pathlib import Path
import re
    
import utils


def main():
    
    # for kaiser
    b1 = 1
    tag_b1 = f'_b{b1}'

    #tag_params = '_p3_n500'
    tag_params = '_p5_n10000'
    tag_mocks = tag_params
    dir_mocks = f'/scratch/kstoreyf/muchisimocks/muchisimocks_lib{tag_mocks}'
    fn_kaiser = f'{dir_mocks}/kaiser_boosts{tag_params}{tag_b1}.csv'

    fn_params = f'{dir_mocks}/params_lh{tag_params}.txt'
    fn_params_fixed = f'{dir_mocks}/params_fixed{tag_params}.txt'
    params_df = pd.read_csv(fn_params, index_col=0)
    param_dict_fixed = pd.read_csv(fn_params_fixed).loc[0].to_dict()
    
    idxs_LH = params_df.index
    
    #compute_kaiser_boosts(idxs_LH, params_df, b1, fn_kaiser)
    compute_pnn_emus(idxs_LH, params_df, param_dict_fixed, tag_mocks)
    
    
    
def compute_kaiser_boosts(idxs_LH, params_df, b1, fn_kaiser):

    # TO TEST
    #idxs_LH = idxs_LH[:10]

    expfactor = 1.0
    kaiser_boosts = []
    for i, idx_LH in enumerate(idxs_LH):
        if i % 100 == 0:
            print(f"Computing kaiser boost for idx_LH {idx_LH}", flush=True)
        param_dict = params_df.loc[idx_LH].to_dict()
        cosmo = utils.get_cosmo(param_dict)
        # kaiser boost only depends on k if neutrino mass is nonzero
        # for kaiser boost, the bias is the linear EULERIAN bias!
        # which is b1_eul = 1 + b1_lag
        bias_eulerian = 1 + b1
        kaiser_boosts.append( cosmo.Kaiser_boost(expfactor, l=0, bias=bias_eulerian) )

    df = pd.DataFrame({'idx_LH': idxs_LH, 'kaiser_boost': kaiser_boosts})
    df.to_csv(fn_kaiser, index=False)
    print(f"Save kaiser boosts to {fn_kaiser}")


def compute_pnn_emus(idxs_LH, params_df, param_dict_fixed, tag_mocks):

    # TO TEST
    #idxs_LH = idxs_LH[:1]

    dir_pnns = f'../data/pnns_mlib/pnns_emu{tag_mocks}'
    Path.mkdir(Path(dir_pnns), parents=True, exist_ok=True)

    # grab k from the first p(k)
    tag_pk = '_b1000'
    dir_pks = f'../data/pks_mlib/pks{tag_mocks}{tag_pk}'
    idxs_LH_all = np.array([int(re.search(r'pk_(\d+)\.npy', f).group(1)) for f in glob.glob(f'{dir_pks}/pk_*.npy')])
    assert len(idxs_LH_all)>1, "Need at least one computed Pk from data to get k's"
    idx_LH = idxs_LH_all[0]
    fn_pk = f'{dir_pks}/pk_{idx_LH}.npy'        
    pk_obj = np.load(fn_pk, allow_pickle=True).item()
    k = pk_obj['k']
    
    # load emu
    dir_emus_lbias = '/home/kstoreyf/external'
    emu, emu_bounds, emu_param_names = utils.load_emu(dir_emus_lbias=dir_emus_lbias)

    # overall quantities
    for i, idx_LH in enumerate(idxs_LH):
        fn_pnn = f'{dir_pnns}/pnn_{idx_LH}.npy'
        if i % 100 == 0:
            print(f"Computing Pnn {idx_LH}, saving to {fn_pnn}", flush=True)
        param_dict = params_df.loc[idx_LH].to_dict()
        param_dict.update(param_dict_fixed)
        cosmo_params_emu = utils.get_cosmo_emu(param_dict)
        # pnn instead of pk
        k, pnn = emu.get_nonlinear_pnn(k=k, **cosmo_params_emu)
        results = {'k': k, 'pnn': pnn}
        np.save(fn_pnn, results)
        
    print("Done!")




if __name__=='__main__':
    main()