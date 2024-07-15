import os
import numpy as np
from pathlib import Path
import pyfftw
import time

import bacco
import bacco.probabilistic_bias as pb

import utils


os.environ["MKL_SERVICE_FORCE_INTEL"] = str(1)


def main():
    n_grid = 512
    n_grid_target = 128

    tab_mocks = ''
    box_size = 1000.
    n_threads_bacco = 12
    # uses 12 GB, indep of number of threads
    # 12 threads takes ~9 minutes for each map2map run (pos, vel)
    # changing n_threads_m2m doesn't seem to do much of anything
    n_threads_m2m = 4
    deconvolve_lr_field = True
    
    save_intermeds = False
    save_hr_field = False
    run_zspace = False
    
    overwrite_m2m_preds = False
    #overwrite_ZA_fields = False
    
    idx_LH_start = 100
    idx_LH_end = 500
    #idxs = range(idx_LH_start, idx_LH_end)
    #idxs = np.arange(idx_LH_start, idx_LH_end)
    idxs = np.arange(idx_LH_start, idx_LH_end, 2)
    
    #previously was '.'; only works if running script from m2m dir
    dir_m2m = '/dipc/kstoreyf/external/map2map_emu'
    #dir_mocks = f'/dipc/kstoreyf/muchisimocks/data/cosmolib{tab_mocks}'
    dir_mocks = f'/cosmos_storage/cosmosims/muchisimocks_lib{tab_mocks}'
    # for now, copying cosmology from the main cosmolib; TODO change this later!
    dir_cosmopars = '/dipc/kstoreyf/muchisimocks/data/cosmolib'    
    
    bacco.configuration.update({'pk':{'boltzmann_solver': 'CLASS'}})
    bacco.configuration.update({'pknbody' : {'ngrid'  :  n_grid}})
    bacco.configuration.update({'scaling' : {'disp_ngrid' : n_grid}})
    bacco.configuration.update({'number_of_threads': n_threads_bacco})

    #tab_mocks = '_FixedPk'
    if 'FixedPk' in tab_mocks:
        FixedInitialAmplitude = True
    else:
        FixedInitialAmplitude = False
    
    for idx_LH in idxs:
        
        print(f"Starting LH {idx_LH}")
        start = time.time()
        timenow = start
        
        dir_LH = f'{dir_mocks}/LH{idx_LH}'
        print("dir_LH:", dir_LH)
        Path.mkdir(Path(dir_LH), parents=True, exist_ok=True)
            
        # get cosmology
        fn_cosmopars_orig = f'{dir_cosmopars}/LH{idx_LH}/cosmo_{idx_LH}.txt'
        fn_cosmopars = f'{dir_LH}/cosmo_{idx_LH}.txt'
        os.system(f'cp {fn_cosmopars_orig} {fn_cosmopars}')
        cospars = np.loadtxt(fn_cosmopars)
        Omega0, sigma8, HubbleParam, OmegaBaryon,ns, Seed = cospars
        Seed = int(Seed)                

        ## Start cosmology class
        param_dict = {'omega_m':Omega0, 
                'omega_baryon':OmegaBaryon, 
                'hubble':HubbleParam, 
                'neutrino_mass':0.0, 
                'sigma8':sigma8, 
                'ns':ns}
        cosmo = utils.get_cosmo(param_dict, a_scale=1.0, sim_name='quijote')
        print(cosmo)
    
        # We also need the parameters in this order in a text file for m2m
        pars_arr = np.array([Omega0, OmegaBaryon, HubbleParam, ns, sigma8])
        np.savetxt(f'{dir_LH}/cosmo_pars_m2m.txt', pars_arr.T)


        # CREATE A ZA SIMULATION
        fn_ZA_disp = f'{dir_LH}/ZA_disp.npy'
        fn_lin = f'{dir_LH}/lin_field.npy'
        fn_ZA_vel = f'{dir_LH}/ZA_vel.npy'
        
        # need the sim object later on, so for now let's recreate even if have
        # the ZA fields saved
        # if overwrite_ZA_fields \
        #     or not os.path.exists(fn_ZA_disp) \
        #     or not os.path.exists(fn_lin) \
        #     or (run_zspace and not os.path.exists(fn_ZA_vel)):
        print("Generating ZA sim")
        sim, disp_field = bacco.utils.create_lpt_simulation(cosmo, box_size, Nmesh=n_grid, Seed=Seed,
                                                            FixedInitialAmplitude=FixedInitialAmplitude,InitialPhase=0, 
                                                            expfactor=1, LPT_order=1, order_by_order=None,
                                                            phase_type=1, ngenic_phases=True, return_disp=True, 
                                                            sphere_mode=0)
        timeprev = timenow
        timenow = time.time()
        print(f"time to create lpt sim: {timenow-timeprev} s")

        print("Saving sims and params")

        np.save(fn_ZA_disp, disp_field, allow_pickle=True)
        norm=n_grid**3.
        np.save(fn_lin, sim.linear_field[0]*norm,allow_pickle=True)
        if run_zspace:
            np.save(fn_ZA_vel, sim.sdm['vel'].reshape((-1,n_grid,n_grid,n_grid)), allow_pickle=True)

        # RUN MAP2MAP
        fn_disp = f'{dir_LH}/pred_disp.npy'
        if overwrite_m2m_preds or not os.path.exists(fn_disp):
            print("Running map2map")
            ## Positions
            # if i don't pass num-threads, it tries to read from slurm, but i'm not running w slurm
            os.system(f'python {dir_m2m}/m2m.py test --num-threads {n_threads_m2m} ' 
                    f'--test-in-patterns "{fn_ZA_disp}" ' 
                    f'--test-tgt-patterns "{fn_ZA_disp}" '
                    '--in-norms "cosmology.dis" --tgt-norms "cosmology.dis" '
                    '--crop 128 --crop-step 128 --pad 48 '
                    '--model d2d.StyledVNet --batches 1 --loader-workers 7 '
                    f'--load-state "{dir_m2m}/map2map/weights/d2d_weights.pt" '
                    f'--callback-at "{dir_m2m}" ' 
                    f'--test-style-pattern "{dir_LH}/cosmo_pars_m2m.txt"')
            os.system(f'mv ._out.npy {fn_disp}')
            
            timeprev = timenow
            timenow = time.time()
            print(f"time for running map2map positions: {timenow-timeprev} s")

        fn_vel = f'{dir_LH}/pred_vel.npy'
        if overwrite_m2m_preds or not os.path.exists(fn_vel):

            ## Velocities (for RSD)
            if run_zspace:
                os.system(f'python {dir_m2m}/m2m.py test --num-threads {n_threads_m2m} ' 
                    f'--test-in-patterns "{fn_ZA_vel}" ' 
                    f'--test-tgt-patterns "{fn_ZA_vel}" '
                    '--in-norms "cosmology.vel" --tgt-norms "cosmology.vel" '
                    '--crop 128 --crop-step 128 --pad 48 '
                    '--model d2d.StyledVNet --batches 1 --loader-workers 7 '
                    f'--load-state "{dir_m2m}/map2map/weights/v2halov_weights.pt" '
                    f'--callback-at "{dir_m2m}" ' 
                    f'--test-style-pattern "{dir_LH}/cosmo_pars_m2m.txt"')
                os.system(f'mv ._out.npy {fn_vel}')

                timeprev = timenow
                timenow = time.time()
                print(f"time for running map2map velocities: {timenow-timeprev} s")
                            
        # COMPUTE BIAS MODEL
        print("Reloading map2map result")
        ## Read displacement, velocities and linear density
        pred_disp = np.load(fn_disp)
        dens_lin = np.load(f'{dir_LH}/lin_field.npy')

        ## Create regular grid and displace particles
        print("Generating grid")
        grid = bacco.visualization.uniform_grid(npix=n_grid, L=box_size, ndim=3, bounds=False)

        print("Adding predicted displacements")
        pred_pos = bacco.scaler.add_displacement(None,
                                        pred_disp,
                                        box=box_size,
                                        pos=grid.reshape(-1,3),
                                        vel=None,
                                        vel_factor=0,
                                        verbose=True)[0]
        timeprev = timenow
        timenow = time.time()
        print(f"time for adding displacements: {timenow-timeprev} s")

        print("Generating bias fields from particle positions")
        #k_nyq = np.pi * n_grid / box_size
        damping_scale = 0.7 #k_nyq
        predicted_positions_to_bias_fields(n_grid, n_grid_target, box_size, sim, 
                                           dens_lin, pred_pos, damping_scale,
                                           save_hr_field, deconvolve_lr_field, save_intermeds,
                                           dir_LH, idx_LH, tag_bfields='')

        ## Include RSD
        if run_zspace:
            velocities = fv2bro(np.load(fn_vel)).copy(order='C')
            pred_pos_zspace = bacco.statistics.compute_zsd(pred_pos, velocities, 
                                                           cosmo, box_size, zspace_axis=2)
            predicted_positions_to_bias_fields(n_grid, n_grid_target, box_size, sim, 
                                           dens_lin, pred_pos_zspace, damping_scale,
                                           save_hr_field, deconvolve_lr_field, save_intermeds,
                                           dir_LH, idx_LH, tag_bfields='_zspace')
            
        # clean up intermediate data products
        if not save_intermeds:
            fns_to_remove = [fn_disp, fn_vel, fn_ZA_disp, fn_ZA_vel, fn_lin]
            for fn_to_remove in fns_to_remove:
                if os.path.isfile(fn_to_remove):
                    os.system(f'rm {fn_to_remove}')   
            
        timenow = time.time()
        print(f"TOTAL TIME for LH {idx_LH}: {timenow-start} s")

    

def predicted_positions_to_bias_fields(n_grid, n_grid_target, box_size, sim, 
                                        dens_lin, pred_pos, damping_scale,
                                        save_hr_field, deconvolve_lr_field, save_intermeds,
                                        dir_LH, idx_LH, tag_bfields=''):

    timenow = time.time()
    
    ## Start bias model class
    interlacing = False
    print("Setting up bias model")
    bmodel = bacco.BiasModel(sim=sim, linear_delta=dens_lin, ngrid=n_grid, ngrid1=None, 
                            sdm=False, mode="dm",
                            npart_for_fake_sim=n_grid, damping_scale=damping_scale, 
                            bias_model='expansion', deposit_method="cic", 
                            use_displacement_of_nn=False, interlacing=interlacing, 
                            )

    ## Compute lagrangian fields
    print("Computing lagrangian fields")
    bias_fields = bmodel.bias_terms_lag()

    ## Compute eulerian fields
    print("Computing eulerian fields")
    bias_terms_eul_pred=[]
    for ii in range(0,len(bias_fields)):
        bias_terms_pred = bacco.statistics.compute_mesh(ngrid=n_grid, box=box_size, pos=pred_pos, 
                                mass = (bias_fields[ii]).flatten(), deposit_method='cic', 
                                interlacing=interlacing)
        bias_terms_eul_pred.append(bias_terms_pred)
    bias_terms_eul_pred = np.array(bias_terms_eul_pred)
    # shape was (5, 1, n_grid, n_grid, n_grid); this squeezes out the 1 dimension
    bias_terms_eul_pred = np.squeeze(bias_terms_eul_pred)
    
    if save_hr_field:
        print(f"Saving full eulerian fields {tag_bfields}")
        np.save(f'{dir_LH}/bias_fields_eul{tag_bfields}_hr_{idx_LH}.npy', bias_terms_eul_pred, allow_pickle=True)
    timeprev = timenow
    timenow = time.time()
    print(f"time for making eulerian fields {tag_bfields}: {timenow-timeprev} s")

    print("Cutting k-modes")
    bias_terms_eul_pred_kcut = remove_lowk_modes(bias_terms_eul_pred, box_size, n_grid_target)
    timeprev = timenow
    timenow = time.time()
    print(f"time for cutting fields to certain kmode {tag_bfields}: {timenow-timeprev} s")
    if save_intermeds:
        print(f"Saving k-cut, non-deconvolved eulerian fields {tag_bfields}")
        np.save(f'{dir_LH}/bias_fields_eul{tag_bfields}_{idx_LH}.npy', bias_terms_eul_pred_kcut, allow_pickle=True)
    
    print(f"Deconvolving k-cut bias fields {tag_bfields}")
    # Deconvolve bias fields - deconvolving the field with the low-k modes already removed bc much faster.
    # Showed that it gets essentially same result as deconvolving the HR field and then cutting the low-k-modes
    if deconvolve_lr_field:
        bias_terms_eul_pred_kcut_deconvolved = deconvolve_bias_field(bias_terms_eul_pred_kcut,
                                                                     n_grid)
        # for some reason this pb function turns our float32 array into float64, 
        # convert back before saving
        bias_terms_eul_pred_kcut_deconvolved = bias_terms_eul_pred_kcut_deconvolved.astype(np.float32)
        
        np.save(f'{dir_LH}/bias_fields_eul{tag_bfields}_deconvolved_{idx_LH}.npy', 
                bias_terms_eul_pred_kcut_deconvolved, allow_pickle=True)
        timeprev = timenow
        timenow = time.time()
        print(f"time for deconvolving k-cut fields {tag_bfields}: {timenow-timeprev} s")


def deconvolve_bias_field(bias_terms, n_grid_orig):
    # Have to do this in a loop, can't just pass bias_terms_eul_hr to the pb function - maybe could rewrite it, but rn no 
    bias_terms_eul_deconvolved = []
    for bias_term in bias_terms:
        bias_term_deconvolved = pb.convolve_linear_interpolation_kernel(bias_term, 
                                                                        npix=n_grid_orig, mode="deconvolve")
        bias_terms_eul_deconvolved.append(bias_term_deconvolved)
    bias_terms_eul_deconvolved = np.array(bias_terms_eul_deconvolved)
    return bias_terms_eul_deconvolved


def remove_lowk_modes(bias_terms_eul_pred, box_size, n_grid_target):
    # updated to squeeze out the extra dim and then alter where needed
    #bias_terms_eul_pred = np.squeeze(bias_terms_eul_pred)
    n_grid = bias_terms_eul_pred.shape[-1]
    k_nyq = np.pi/box_size*n_grid_target
    kmesh = bacco.visualization.np_get_kmesh( (n_grid, n_grid, n_grid), box_size, real=True)
    mask = (kmesh[:,:,:,0]<=k_nyq) & (kmesh[:,:,:,1]<=k_nyq) & (kmesh[:,:,:,2]<=k_nyq) & (kmesh[:,:,:,0]>-k_nyq) & (kmesh[:,:,:,1]>-k_nyq) & (kmesh[:,:,:,2]>-k_nyq)
    bias_terms_eul_pred_kcut=[]
    assert n_grid_target%2==0, "n_grid_target must be even!"
    for fid in range(5):
        field = bias_terms_eul_pred[fid]
        deltak = pyfftw.builders.rfftn(field, auto_align_input=False, auto_contiguous=False, avoid_copy=True)
        deltakcut = deltak()[mask]
        deltakcut= deltakcut.reshape(n_grid_target, n_grid_target, int(n_grid_target/2)+1)
        delta = pyfftw.builders.irfftn(deltakcut, axes=(0,1,2))()
        bias_terms_eul_pred_kcut.append(delta)
    bias_terms_eul_pred_kcut = np.array(bias_terms_eul_pred_kcut)
    return bias_terms_eul_pred_kcut


def fv2bro(t_fv_field) :
    '''Returns back row ordered array (shape n_grid * n_grid * n_grid, 3) from front vector (shape 3, n_grid, n_grid, n_grid)'''
    return np.reshape(t_fv_field, (3, int(t_fv_field.size / 3))).T



if __name__=='__main__':
    main()