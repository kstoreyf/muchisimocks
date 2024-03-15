import numpy as np 


param_label_dict = {'omega_cold': r'$\Omega_\mathrm{m}$',
                'sigma8_cold': r'$\sigma_{8}$',
                'sigma_8': r'$\sigma_{8}$',
                'hubble': r'$h$',
                'h': r'$h$',
                'ns': r'$n_\mathrm{s}$',
                'n_s': r'$n_\mathrm{s}$',
                'omega_baryon': r'$\Omega_\mathrm{b}$',}


def idxs_train_val_test(random_ints, frac_train=0.70, frac_val=0.15, frac_test=0.15):

    tol = 1e-6
    assert abs((frac_train+frac_val+frac_test) - 1.0) < tol, "Fractions must add to 1!" 
    N_halos = len(random_ints)
    int_train = int(frac_train*N_halos)
    int_test = int((1-frac_test)*N_halos)

    idxs_train = np.where(random_ints < int_train)[0]
    idxs_test = np.where(random_ints >= int_test)[0]
    idxs_val = np.where((random_ints >= int_train) & (random_ints < int_test))[0]

    return idxs_train, idxs_val, idxs_test


def split_train_val_test(arr, idxs_train, idxs_val, idxs_test):
    arr_train = arr[idxs_train]
    arr_val = arr[idxs_val]
    arr_test = arr[idxs_test]
    return arr_train, arr_val, arr_test