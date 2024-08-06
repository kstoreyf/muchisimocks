import itertools
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
#import os
import pandas as pd
import time

import bacco

import sys
sys.path.append('/dipc/kstoreyf/muchisimocks/scripts')
import utils
# import plotter
# import data_creation_pipeline as dcp
# import compute_biased_pks_fields as cpk


def main():
    start = time.time()

    n_grid = 512
    #n_grid = 64
    tag_extra = '_posfromweb'

    #sim_type = 'bacco'
    sim_type = 'quijote'
    
    if sim_type == 'bacco':
        sim_name = 'TheOne_N1536_L512'
        dens_lin, pos, cosmo, box_size = load_data_bacco(sim_name, n_grid)
    elif sim_type == 'quijote':
        idx_LH_str = '0663'
        sim_name = f'quijote_LH{idx_LH_str}'
        dens_lin, pos, cosmo, box_size = load_data_quijote(idx_LH_str)

    compute_pnn_pipeline(dens_lin, pos, cosmo, box_size, n_grid, sim_name, tag_extra)
    
    print(f"Done! Time: {time.time()-start} s")

    
def load_data_bacco(sim_name, n_grid):
    
    print("Loading bacco data...")
    if sim_name=='TheOne_N1536_L512':
        basedir = "/cosmos_storage/cosmosims/MultiCosmology_N1536/power_N1536_L512.0_output/0.00"
    snapnum_init = 0
    halo_file_init = f"groups_{snapnum_init:03}/fof_subhalo_history_tab_orph_wweight_{snapnum_init:03}"
    sim = bacco.Simulation(basedir=basedir, 
                        halo_file=halo_file_init,
                        )
    
    expfactor_target = 1.0
    diff = expfactor_target-sim.snaplist['a']
    snapnum = sim.snaplist['snap'][np.argmin(diff[diff>0])]
    expfactor = sim.snaplist['a'][np.argmin(diff[diff>0])]
    print(snapnum, expfactor)
                             
    from bacco.cosmo_parameters import TheOne as TheOne_dict   
    cosmo_theone = bacco.Cosmology(**TheOne_dict, expfactor=expfactor)

    halo_file = f"groups_{snapnum:03}/fof_subhalo_history_tab_orph_wweight_{snapnum:03}"
    print(halo_file)

    sim = bacco.Simulation(basedir=basedir, 
                        sim_cosmology=cosmo_theone, 
                        halo_file=halo_file,
                        )

    # only need dens_lin and pos in the end!
    box_size = sim.header['BoxSize']
    # double check that this is what we want!
    dens_lin = sim.get_linear_field(ngrid=n_grid, quantity='delta')
    print('dens_lin.shape:', dens_lin.shape)
    # can do more directly from bmodel, but it just calls this
    # ind = bmodel.index_from_lag_neighbour('sdm')
    
    # GET POS
    regular_grid = bacco.visualization.uniform_grid(npix=n_grid, L=box_size, ndim=3, bounds=False)
    idstart = 1 # the default for bmodel
    ind = sim.find_lagrangian_neighbour(regular_grid, mode='sdm', return_tree=False,idstart=idstart)
    ind = ind.reshape(-1)
    pos = sim.sdm['pos'][ind]
    print('pos.shape:', pos.shape)

    return dens_lin, pos, cosmo_theone, box_size



def load_data_quijote(idx_LH_str):
    
    print("Loading quijote data...")

    #data_source = 'marcos'
    data_source = 'website'

    dir_data = '/cosmos_storage/home/mpelle/Yin_data/Quijote'
    param_names = ['omega_m', 'omega_baryon', 'h', 'n_s', 'sigma_8']

    box_size = 1000.0
    fn_params = f'{dir_data}/LH{idx_LH_str}/param_{idx_LH_str}.txt'
    param_vals = np.loadtxt(fn_params)
    param_dict = dict(zip(param_names, param_vals))
    cosmo_quijote = utils.get_cosmo(param_dict)
    
    fn_dens_lin = f'{dir_data}/LH{idx_LH_str}/lin_den_{idx_LH_str}.npy'
    dens_lin = np.load(fn_dens_lin)[0]
    print('dens_lin.shape:', dens_lin.shape)

    if data_source == 'marcos':
        fn_disp = f'{dir_data}/LH{idx_LH_str}/dis_{idx_LH_str}.npy'
        disp = np.load(fn_disp) # sim
        n_grid = disp.shape[-1]

        grid = bacco.visualization.uniform_grid(npix=n_grid, L=box_size, ndim=3, bounds=False)

        pos = bacco.scaler.add_displacement(None,
                                        disp,
                                        box=box_size,
                                        pos=grid.reshape(-1,3),
                                        vel=None,
                                        vel_factor=0,
                                        verbose=False)[0]
        
    elif data_source == 'website':
        
        import readgadget
        snapshot = '/dipc/kstoreyf/Quijote_simulations/Snapshots/latin_hypercube/663/snapdir_004/snap_004'
        ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
        # read positions, velocities and IDs of the particles
        pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
        
    print('pos.shape:', pos.shape)

    return dens_lin, pos, cosmo_quijote, box_size



def compute_pnn_pipeline(dens_lin, pos, cosmo, box_size, n_grid, sim_name, tag_extra=''):

    print("Setting up bias model")
    bmodel = bacco.BiasModel(sim=None, linear_delta=dens_lin, ngrid=n_grid, ngrid1=None,
                            sdm=False, mode="dm", # these are the defaults - but do we need to change for bacco?
                            BoxSize=box_size,
                            npart_for_fake_sim=n_grid, #damping_scale=damping_scale,
                            bias_model='expansion', deposit_method="cic",
                            use_displacement_of_nn=False, interlacing=False,
                            )

    print("Computing lagrangian fields")
    bias_fields = bmodel.bias_terms_lag()

    # Emulator creation method
    #print("running bias model")
    # bmodel = bacco.BiasModel(sim=sim, ngrid=n_grid, sdm=True, mode="sdm", mode_vel="combine",
    #                         npart_for_fake_sim=n_grid, damping_scale=0.75, bias_model='expansion', mean_num_dens=None,
    #                         stochastic=False, deposit_method="cic", use_RSD=redshift_space, use_displacement_of_nn=False, 
    #                         interlacing=False,
    #                         indices=None, indices_vel=None, sdmhids=None)

    # print("Computing power terms")
    # power_all_terms = bmodel.compute_power_terms(kmin=0.01, kmax=1, nbins=30, log_binning=True)

    print("Computing eulerian fields")
    bias_terms_eul=[]
    for ii in range(0,len(bias_fields)):
        bias_terms = bacco.statistics.compute_mesh(ngrid=n_grid, box=box_size, pos=pos, 
                                mass = (bias_fields[ii]).flatten(), deposit_method='cic', 
                                interlacing=False)
        bias_terms_eul.append(bias_terms)
    bias_terms_eul = np.array(bias_terms_eul)
    # shape was (5, 1, n_grid, n_grid, n_grid); this squeezes out the 1 dimension
    bias_terms_eul = np.squeeze(bias_terms_eul)

    print("Computing the 15 PNN cross power spectra")
    norm=bmodel.npart
    normalise_grid = False
    deconvolve_grid = True
    k_min = 0.01
    k_max = 0.68
    n_bins = 30
    log_binning = True
    correct_grid = True

    args_power = {
                    'interlacing':bmodel.interlacing,
                    'kmin':k_min,
                    'kmax':k_max,
                    'nbins':n_bins,
                    'log_binning':log_binning,
                    'deposit_method':'cic',
                    'correct_grid': correct_grid,
                    'compute_correlation':False,
                    'zspace':False, #we include the velocities before
                    'compute_power2d':False}

    prod = np.array(list(itertools.combinations_with_replacement(np.arange(len(bias_terms_eul)),r=2)))

    if(correct_grid):
        lt_k = np.logspace(np.log10(np.pi / bmodel.BoxSize), np.log10(2 * np.pi / bmodel.BoxSize * bmodel.ngrid1), num=100)
        pk_lpt = bacco.utils.compute_pt_15_basis_terms(cosmo, expfactor=cosmo.expfactor, wavemode=lt_k)
        #pk_lpt = bmodel.compute_pt_15_basis_terms(cosmo_theone, expfactor=cosmo_theone.expfactor, wavemode=lt_k)

    power_all_terms = []
    for ii in range(0,len(prod)):
        print(ii)
        if(correct_grid):
            if ii in [1,5,9,12]:
                pk_lt = {'k':lt_k, 'pk':pk_lpt[0][ii], 'pk_nlin':pk_lpt[0][ii], 'pk_lt_log': True}
            if ii in [2,3,4,7,8,11,13]:
                pk_lt = {'k':lt_k, 'pk':pk_lpt[0][ii], 'pk_nlin':pk_lpt[0][ii], 'pk_lt_log': False}
            else:
                pk_lt = None
            args_power['correct_grid'] = False if ii == 11 else correct_grid

        else:
            pk_lt = None
            
        jack_error = False
        n_jack = 0
        deconvolve_grid = True
        power_term = bacco.statistics.compute_crossspectrum_twogrids(grid1=bias_terms_eul[prod[ii,0]]/norm,
                                                        grid2=bias_terms_eul[prod[ii,1]]/norm,
                                                        normalise_grid1=normalise_grid,
                                                        normalise_grid2=normalise_grid,
                                                        deconvolve_grid1=deconvolve_grid,
                                                        deconvolve_grid2=deconvolve_grid,
                                                        cosmology=cosmo,
                                                        ngrid=n_grid,
                                                        box=box_size,
                                                        pk_lt = pk_lt,jack_error=jack_error,
                                                        n_jack=n_jack,
                                                        **args_power)
        power_all_terms.append(power_term)

    fn_pnn = f'../data/pnns/pnn_{sim_name}{tag_extra}.npy'
    print("Saving to", fn_pnn)
    np.save(fn_pnn, power_all_terms)

    print("Compute power spectra of lag fields")
    pk_objs_lag = []
    for i in range(len(bias_fields)):
        pk_obj_lag = bacco.statistics.compute_crossspectrum_twogrids(
                        grid1=bias_fields[i]/norm,
                        grid2=bias_fields[i]/norm,
                        cosmology=cosmo,
                        ngrid=n_grid,
                        box=box_size,
                        normalise_grid1=normalise_grid,
                        normalise_grid2=normalise_grid,
                        deconvolve_grid1=deconvolve_grid,
                        deconvolve_grid2=deconvolve_grid,
                        **args_power)
        pk_objs_lag.append(pk_obj_lag)
    
    fn_pk_lag = f'../data/pks/pks_lagrangian_{sim_name}{tag_extra}.npy'
    print("Saving to", fn_pk_lag)
    np.save(fn_pk_lag, pk_objs_lag)  
        


if __name__=='__main__':
    main()