import itertools
import numpy as np
import math
import matplotlib as mpl
from matplotlib import pyplot as plt
#import os
import pandas as pd
import pickle
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
    tag_extra = f'_createlpt_posfromweb_ngrid{n_grid}'
    #tag_extra = '_createlpt_posmarcos'
    #tag_extra = '_fromweb_idcorr'
    #tag_extra = f'_ngrid{n_grid}'

    #sim_type = 'bacco'
    sim_type = 'quijote'
    
    if sim_type == 'bacco':
        sim_name = 'TheOne_N1536_L512'
        dens_lin, pos, cosmo, box_size = load_data_bacco(sim_name, n_grid=n_grid)
    elif sim_type == 'quijote':
        idx_LH_str = '0663'
        sim_name = f'quijote_LH{idx_LH_str}'
        dens_lin, pos, cosmo, box_size = load_data_quijote(idx_LH_str, n_grid=n_grid)

    compute_pnn_pipeline(dens_lin, pos, cosmo, box_size, n_grid, sim_name, tag_extra)
    
    print(f"Done! Time: {time.time()-start} s")

    
def load_data_bacco(sim_name, n_grid=512):
    
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



def load_data_quijote(idx_LH_str, n_grid=512):
    
    print("Loading quijote data...")
    #n_grid = 512 
    
    #source_dens_lin = 'marcos'
    source_dens_lin = 'create_lpt'
    #source_dens_lin = 'website'
    
    #source_pos = 'create_lpt'
    #source_pos = 'marcos'
    source_pos = 'website'
    
    if source_dens_lin=='marcos' or source_pos=='marcos':
        # maybe always needs to be??
        assert n_grid==512, "n_grid should be 512 for marcos fields"

    dir_data = '/cosmos_storage/home/mpelle/Yin_data/Quijote'
    param_names = ['omega_m', 'omega_baryon', 'h', 'n_s', 'sigma_8']

    box_size = 1000.0
    fn_params = f'{dir_data}/LH{idx_LH_str}/param_{idx_LH_str}.txt'
    param_vals = np.loadtxt(fn_params)
    param_dict = dict(zip(param_names, param_vals))
    cosmo_quijote = utils.get_cosmo(param_dict)

    sim = None
    if source_dens_lin == 'marcos':    
        fn_dens_lin = f'{dir_data}/LH{idx_LH_str}/lin_den_{idx_LH_str}.npy'
        dens_lin = np.load(fn_dens_lin)[0]
        print('dens_lin.shape:', dens_lin.shape)

    elif source_dens_lin == 'create_lpt':
        ngenic_phases = False
        phase_type = 0

        seed = int(idx_LH_str)
        expfactor = 1.0
        FixedInitialAmplitude = False

        sim = bacco.utils.create_lpt_simulation(cosmo_quijote, box_size, Nmesh=n_grid, Seed=seed,
                                                            FixedInitialAmplitude=FixedInitialAmplitude,InitialPhase=0, 
                                                            expfactor=expfactor, LPT_order=2, order_by_order=None,
                                                            phase_type=phase_type, ngenic_phases=ngenic_phases, 
                                                            return_disp=False, sphere_mode=0)
        
        dens_lin = sim.get_linear_field(ngrid=n_grid, quantity='delta')

    # elif source_dens_lin == 'website':
        
    #     import readgadget

    #     # input files
    #     snapshot_ics = '/dipc/kstoreyf/Quijote_simulations/Snapshots/latin_hypercube/663/ICs/ics'
    #     ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)

    #     # read positions, velocities and IDs of the particles
    #     pos_ics_raw = readgadget.read_block(snapshot_ics, "POS ", ptype)/1e3 #positions in Mpc/h
    #     #ids_ics = readgadget.read_block(snapshot_ics, "ID  ", ptype)-1   #IDs starting from 0
        
    #     n_grid = 512
    #     pos_ics_mesh = bacco.statistics.compute_mesh(ngrid=n_grid, box=box_size, pos=pos_ics_raw, 
    #                                             deposit_method='cic', interlacing=False)
    #     pos_ics_mesh = np.squeeze(pos_ics_mesh)
        
    #     dens_lin_ics_mesh = pos_ics_mesh/np.mean(pos_ics_mesh) - 1.0
        
    #     # tranform to z=0
    #     dens_lin = dens_lin_ics_mesh / cosmo_quijote.get_growth_z(1/(1+127))

    else:
        raise ValueError(f"source_dens_lin not recognized!")


    if source_pos == 'marcos':
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
    
    # elif source_pos == 'create_lpt':
    #     if disp_fromlpt is None:
    #         raise ValueError("Must also have source_dens_lin=='create_lpt'!")
        
    #     grid = bacco.visualization.uniform_grid(npix=n_grid, L=box_size, ndim=3, bounds=False)

    #     pos = bacco.scaler.add_displacement(None,
    #                                     disp_fromlpt,
    #                                     box=box_size,
    #                                     pos=grid.reshape(-1,3),
    #                                     vel=None,
    #                                     vel_factor=0,
    #                                     verbose=False)[0]
        
    elif source_pos == 'website':
        
        import readgadget
        snapshot = '/dipc/kstoreyf/Quijote_simulations/Snapshots/latin_hypercube/663/snapdir_004/snap_004'
        ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
        # read positions, velocities and IDs of the particles
        pos_raw = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
        
        idx_LH = int(idx_LH_str)
        fn_lag_index = f"/cosmos_storage/data_sharing/quijote_{idx_LH}_neighfile.pickle"
        with open(fn_lag_index, 'rb') as f:
            lag_index = pickle.load(f)
        print(lag_index.shape)
        
        pos = pos_raw[lag_index]
    
    else: 
        raise ValueError(f"source_pos not recognized")
        
    print('pos.shape:', pos.shape)

    return dens_lin, pos, cosmo_quijote, box_size



def compute_pnn_pipeline(dens_lin, pos, cosmo, box_size, n_grid, sim_name, tag_extra=''):

    print("Setting up bias model")
    print(len(pos), len(pos)**(1/3))
    n_grid_frompos = math.ceil(len(pos)**(1/3))# dens_lin.shape[-1] # or should get this from particles?
    print('n_grid', n_grid)
    print('n_grid_frompos', n_grid_frompos)
    print(dens_lin.shape)
    #n_grid_bfields = 64

    # ok i do feel like all of these need to line up?
    # wait lets try npart_for_fake_sim
    npart_for_fake_sim = 64
    bmodel = bacco.BiasModel(sim=None, linear_delta=dens_lin, ngrid=n_grid_frompos, ngrid1=None,
                            sdm=False, mode="dm", # these are the defaults - but do we need to change for bacco?
                            BoxSize=box_size,
                            npart_for_fake_sim=npart_for_fake_sim, #damping_scale=damping_scale,
                            bias_model='expansion', deposit_method="cic",
                            use_displacement_of_nn=False, interlacing=False,
                            )


    ### adding this sec to test 
    # lin_field = bmodel.linear_field
    # print(lin_field.shape)
    # bias_term_lin = bacco.statistics.compute_mesh(ngrid=n_grid_frompos, box=box_size, pos=pos, 
    #                             mass = (lin_field).flatten(), deposit_method='cic', 
    #                             interlacing=False)
    # print(bias_term_lin.shape)
    # print(kjsdf)

    print("Computing lagrangian fields")
    # these have size n_grid
    bias_fields = bmodel.bias_terms_lag()

    print(bias_fields.shape)
    
    
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
        print(ii)
        print(bias_fields[ii].shape)
        # mass has to have same shape as pos; so bias fields need to 
        # which ngrid here??
        bias_terms = bacco.statistics.compute_mesh(ngrid=n_grid_frompos, box=box_size, pos=pos, 
                                mass = (bias_fields[ii]).flatten(), deposit_method='cic', 
                                interlacing=False)
        print(bias_terms.shape)
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
    jack_error = False
    n_jack = 0
        
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
        print("bmodel.ngrid1", bmodel.ngrid1)
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
            
        print("bias_terms_eul[prod[ii,1]].shape", bias_terms_eul[prod[ii,1]].shape)
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