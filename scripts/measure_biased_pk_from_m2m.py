import numpy as np
import bacco
import os
import copy
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import deepdish as dd

ngrid = 512 #1024 #512 #256 #128 #256 #1400
nmean=0.01
BoxSize=1000

bacco.configuration.update({'pknbody' : {'ngrid'  :  ngrid}})
bacco.configuration.update({'pknbody' : {'log_binning' : True}})
bacco.configuration.update({'pknbody' : {'log_binning_kmax' : 0.99506136}})#
bacco.configuration.update({'pknbody' : {'log_binning_nbins' : 100}})
bacco.configuration.update({'pknbody' : {'min_k' : 0.01721049}})
bacco.configuration.update({'pk' : {'maxk' : 0.99506136}})
bacco.configuration.update({'pknbody' : {'interlacing' : False}})

bacco.configuration.update({'pknbody' : {'depmethod' : 'cic'}})

bacco.configuration.update({'nonlinear' : {'concentration' : 'ludlow16'}})

bacco.configuration.update({'number_of_threads' : 12})
bacco.configuration.update({'scaling' : {'disp_ngrid' : ngrid}})

bacco.configuration.update({'pk':{'boltzmann_solver': 'CLASS'}})

# b1, b2, bs2, blap
biases_vec = [[b1, 0, 0, 0] for b1 in np.linspace(0.5, 1.5, 10)]

grid = bacco.visualization.uniform_grid(npix=ngrid, L=BoxSize, ndim=3, bounds=False)

indicesLH = np.array([10,29,37,40,70,85,127,158,165,184,208,220,240,254,267,274,293,305,336,374,375,388,433,444,
                      464,502,534,542,574,598,605,628,652,663,676,700,702,721,737,762,809,822,825,837,853,864,882,
                      899,901,911,939,948,950,951,964,976,977,1016,1022,1041,1050,1060,1082,1091,1103,1114,1147,
                      1157,1173,1175,1219,1222,1299,1309,1314,1317,1331,1365,1372,1378,1391,1397,1418,1444,1459,
                      1510,1512,1513,1515,1517,1533,1553,1567,1568,1599,1622,1642,1657,1659,1667])

for i in indicesLH:

    pred_disp = np.load('/dipc_storage/mpelle/Yin_data/Quijote/LH%04d/pred_pos_%04d.npy'%(indicesLH[i], indicesLH[i]))

    pred_pos = bacco.scaler.add_displacement(None,
                                pred_disp,
                                box=BoxSize,
                                pos=grid.reshape(-1,3),
                                vel=None,
                                vel_factor=0,
                                verbose=True)[0]

    dens_yin = np.load('/dipc_storage/mpelle/Yin_data/Quijote/LH%04d/lin_den_%04d.npy'%(indicesLH[i], indicesLH[i]))

    k_nyq = np.pi * ngrid / BoxSize

    damping_scale = k_nyq

    bmodel = bacco.BiasModel(sim=None, linear_delta=dens_yin[0], ngrid=ngrid, ngrid1=ngrid,
                            sdm=True, mode="dm", BoxSize=BoxSize,
                            npart_for_fake_sim=ngrid, damping_scale=damping_scale,
                            bias_model='expansion', deposit_method="cic",
                            use_displacement_of_nn=False, interlacing=False,
                            )

    bias_fields = bmodel.bias_terms_lag()

    bias_terms_eul_pred=[]
    for ii in range(0,len(bias_fields)):
        bias_terms_pred = bacco.statistics.compute_mesh(ngrid=ngrid, box=BoxSize, pos=pred_pos,
                                mass = (bias_fields[ii]).flatten(), deposit_method='cic',
                                interlacing=False)
        bias_terms_eul_pred.append(bias_terms_pred)
    bias_terms_eul_pred = np.array(bias_terms_eul_pred)

    a_Quijote = 1
    Ob = 0.049
    Om = 0.3175
    hubble = 0.6711
    ns = 0.9624
    sigma8 = 0.834
    cosmopars = dict(
            omega_cdm=Om-Ob,
            omega_baryon=Ob,
            hubble=hubble,
            ns=ns,
            sigma8=sigma8,
            tau=0.0561,
            A_s=None,
            neutrino_mass=0.,
            w0=-1,
            wa=0,
            tag="cosmo_BOSS"
        )


    cosmo_Quijote = bacco.Cosmology(**cosmopars)
    cosmo_Quijote.set_expfactor(a_Quijote)

    ##################

    # These are the important arguments of the P(k)
    args_power = {'ngrid':ngrid,
                    'box':BoxSize,
                    'cosmology':cosmo_Quijote,
                    'interlacing':False,
                    'kmin':0.01,
                    'kmax':1,
                    'nbins':50,
                    'correct_grid':True,
                    'log_binning':True,
                    'deposit_method':'cic',
                    'compute_correlation':False,
                    'zspace':False,
                    'compute_power2d':False}


    #This is what I was talking about some corrections one can make to the P(k) computation
    #Not very important though, ask Raul if you want to know more
    lt_k = np.logspace(np.log10(np.pi / BoxSize), np.log10(2 * np.pi / BoxSize * ngrid), num=90)
    pk_lpt = bacco.utils.compute_pt_15_basis_terms(cosmo_Quijote, expfactor=cosmo_Quijote.expfactor, wavemode=lt_k)

    #Normalise the grid before P(k) computation
    #one can normalise later too, I chose to do it here
    norm=ngrid**3.
    bias_terms_eul_norm_pred = bias_terms_eul_pred/norm


    #Compute a dummy variable with the 15 combinations of 5 distinct objects
    import itertools
    prod = np.array(list(itertools.combinations_with_replacement(np.arange(bias_terms_eul_pred.shape[0]),r=2)))

    #Compute the P(k) of the 15 terms
    power_all_terms_pred = []
    for ii in range(0,len(prod)):
        pk_lt = {'k':lt_k, 'pk':pk_lpt[0][ii], 'pk_nlin':pk_lpt[0][ii], 'pk_lt_log': True}
        if ii in [2,3,4,7,8,11,13]:
            pk_lt['pk_lt_log'] = False
        args_power['correct_grid'] = False if ii == 11 else True

        power_term_pred = bacco.statistics.compute_crossspectrum_twogrids(grid1=bias_terms_eul_norm_pred[prod[ii,0]],
                                                        grid2=bias_terms_eul_norm_pred[prod[ii,1]],
                                                        normalise_grid1=False,
                                                        normalise_grid2=False,
                                                        deconvolve_grid1=True,
                                                        deconvolve_grid2=True,
                                                        **args_power)
        power_all_terms_pred.append(power_term_pred)


    for ib, biases in enumerate(biases_vec):

        bias_extended = np.concatenate([[1], biases])

        prod = np.array(list(itertools.combinations_with_replacement(np.arange(5), r=2)))

        sum_terms_power = 0
        for ii in range(0,15):
            fac = 2 if prod[ii,0]!=prod[ii,1] else 1
            sum_terms_power += bias_extended[prod[ii,0]] * bias_extended[prod[ii,1]] * power_all_terms_pred[ii]['pk'] * fac

        kk = power_all_terms_pred[0]['k']

        fname = 'biased_pk_m2m_num_%04d_bias_num_%d.txt'%(indicesLH[i], ib)
        np.savetxt(fname, np.transpose([kk, sum_terms_power]))