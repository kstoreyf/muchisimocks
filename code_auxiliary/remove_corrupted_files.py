import numpy as np
import os

import data_loader


# Set to True to see what would be removed without actually removing files
DRY_RUN = False
#stat_name = 'pnn'
stat_name = 'bispec'
tag_params = '_p5_n10000'
tag_biasparams = '_biaszen_p4_n200000'
#tag_params = '_test_p5_n1000'
#tag_biasparams = '_biaszen_p4_n1000'
#tag_params = '_quijote_p0_n1000'
#tag_biasparams = '_b1000_p0_n1'
tag_noise = None
tag_Anoise = None
n = int(tag_params.split('_n')[-1])
print(f"n: {n}")

params_df, param_dict_fixed, biasparams_df, biasparams_dict_fixed, Anoise_df, Anoise_dict_fixed, random_ints, random_ints_bias = data_loader.load_params(tag_params, tag_biasparams, tag_Anoise=tag_Anoise)

dir_statistics = data_loader.get_dir_statistics(stat_name, tag_params, tag_biasparams, 
                                        tag_noise=tag_noise, tag_Anoise=tag_Anoise)
print(f"dir_statistics: {dir_statistics}")

for idx_LH in range(n):
    #fn = f'/scratch/kstoreyf/muchisimocks/data/pnns_mlib/pnns{tag_params}/pnn_{i}.npy'
    #fn = f"{dir_statistics}/{stat_name}_{idx_LH}.npy"
    dir_statistics, fns_statistics, idxs_bias, idxs_noise = data_loader.get_fns_statistic_precomputed(stat_name, idx_LH, dir_statistics,
                                        params_df, biasparams_df, tag_biasparams, tag_Anoise)
    for fn in fns_statistics:
        #print(f"Checking file: {fn}")
        if not os.path.isfile(fn):
            print(f"File does not exist: {fn}")
            continue
        try:
            data = np.load(fn, allow_pickle=True)
            #print(f"Successfully loaded {fn}")
        except Exception as e:
            print(f"Failed to load {fn}: {e}")
            if DRY_RUN:
                print(f"[DRY RUN] Would remove {fn}")
            else:
                try:
                    os.remove(fn)
                    print(f"Removed {fn}")
                except Exception as remove_error:
                    print(f"Failed to remove {fn}: {remove_error}")

if DRY_RUN:
    print("\n[DRY RUN MODE] No files were actually removed. Set DRY_RUN=False to remove corrupted files.")