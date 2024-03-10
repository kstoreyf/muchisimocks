import numpy as np

import bacco

def main():
    tag_emuPk = '_2param'
    kk, Pk = load_emuPks(tag_emuPk)
    box_size = 1000.
    
    fn_emuPkerrG = f'../data/emuPks/emuPks_errgaussian{tag_emuPk}.npy'

    errs_gaussian = []
    for i in range(Pk.shape[0]):
        if i%100==0:
            print(i)
        err_gaussian = bacco.statistics.approx_pk_gaussian_error(kk, Pk[i], box_size)
        errs_gaussian.append(err_gaussian)
    errs_gaussian = np.array(errs_gaussian)
    np.save(fn_emuPkerrG, errs_gaussian)
    
    
def load_emuPks(tag_emuPk):

    fn_emuPk = f'../data/emuPks/emuPks{tag_emuPk}.npy'
    fn_emuk = f'../data/emuPks/emuPks_k{tag_emuPk}.txt'

    Pk = np.load(fn_emuPk)
    #theta = np.genfromtxt(fn_emuPk_params, delimiter=',', names=True)
    #param_names = theta.dtype.names
    # from tuples to 2d array
    #theta = np.array([list(tup) for tup in theta])
    kk = np.genfromtxt(fn_emuk)
    return kk, Pk


if __name__=='__main__':
    main()