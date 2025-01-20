import argparse
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pyfftw
import time

import bacco
import bacco.probabilistic_bias as pb

import utils


pyfftw.config.NUM_THREADS = 8
print("pyfftw nthreads", pyfftw.config.NUM_THREADS)

os.environ["MKL_SERVICE_FORCE_INTEL"] = str(1)
#os.environ["CUDA_VISIBLE_DEVICES"]=""

#######################
# Run with e.g. nohup nice -n 10 python data_creation_pipeline.py 0 &> logs/datagen_LH0.out &
#######################



def main():
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('idx_mock_start', type=int)
    argparser.add_argument('idx_mock_end', type=int, nargs='?', default=None)  
    argparser.add_argument('--modecosmo', action='store', choices=['LH', 'fixed'], default='LH')

    args = argparser.parse_args()
    idx_mock_end = args.idx_mock_end
    if idx_mock_end is None:
        idx_mock_end = args.idx_mock_start + 1
    for idx_mock in range(args.idx_mock_start, idx_mock_end):
        if args.modecosmo == 'LH':
            run_LH(idx_mock)
        if args.modecosmo == 'fixed':
            run_fixedcosmo(idx_mock)
            

def run_LH(idx_mock):
    
    print("Setting up", flush=True)
    n_grid = 512
    n_grid_target = 128

    #tag_params = f'_p3_n50'
    #tag_params = f'_p3_n10'
    #tag_params = f'_p5_n10000'
    tag_params = f'_test_p5_n1000'
    tag_mocks = f'{tag_params}'
    #tag_mocks = f'{tag_params}_timetests'
    box_size = 1000.

    # uses 12 GB, indep of number of threads
    # 1 job: ~200s = ~3 min each for pos / vel
    # 2 jobs: ~400s = ~7 min each for pos / vel
    # 3 jobs: ~400-500s = ~8 min each for pos / vel
    n_workers_m2m = 0 #0 disables multiprocessing
    n_threads_bacco = 8
    print(f"n_workers_m2m = {n_workers_m2m}, n_threads_bacco = {n_threads_bacco}", flush=True)

    #dir_mocks = f'/dipc/kstoreyf/muchisimocks/data/cosmolib{tag_mocks}'
    #dir_mocks = f'/cosmos_storage/cosmosims/muchisimocks/muchisimocks_lib{tag_mocks}'
    dir_mocks = f'/scratch/kstoreyf/muchisimocks/muchisimocks_lib{tag_mocks}'
    # need to make mock dir now so we can move param files into it
    Path.mkdir(Path(dir_mocks), parents=True, exist_ok=True)
    print(f"Made mock dir {dir_mocks}", flush=True)
    
    # Deal with cosmological parameters
    
    # TODO update so can handle fixedcosmo case in this same framework
    
    dir_params = '../data/params'
    fn_params_orig = f'{dir_params}/params_lh{tag_params}.txt'
    fn_params_fixed_orig = f'{dir_params}/params_fixed{tag_params}.txt'
    # copy parameters to mocks folder to make sure we know which params were used
    fn_params = f'{dir_mocks}/params_lh{tag_params}.txt'
    fn_params_fixed = f'{dir_mocks}/params_fixed{tag_params}.txt'
    if not os.path.exists(fn_params):
        os.system(f'cp {fn_params_orig} {fn_params}')
    if not os.path.exists(fn_params_fixed):
        os.system(f'cp {fn_params_fixed_orig} {fn_params_fixed}')
    params_df = pd.read_csv(fn_params, index_col=0)
    param_dict_fixed = pd.read_csv(fn_params_fixed).loc[0].to_dict()
    
    param_dict = params_df.loc[idx_mock].to_dict()
    param_dict.update(param_dict_fixed)
    seed = idx_mock
    print("param_dict:", param_dict, flush=True)
    
    run_single(param_dict, seed, idx_mock, dir_mocks,
               n_grid=n_grid, n_grid_target=n_grid_target, box_size=box_size,
               n_threads_bacco=n_threads_bacco, n_workers_m2m=n_workers_m2m)
    

def run_fixedcosmo(idx_mock):
    
    print("Setting up", flush=True)
    n_grid = 512
    n_grid_target = 128

    tag_mocks = f'_fixedcosmo'
    box_size = 1000.

    # NOTE: when cuda for pytorch isn't working, map2map just doesn't 
    # really run nor produce an output file! make sure cuda is available  
    n_workers_m2m = 0 #0 disables multiprocessing
    n_threads_bacco = 8
    print(f"n_workers_m2m = {n_workers_m2m}, n_threads_bacco = {n_threads_bacco}", flush=True)

    dir_mocks = f'/scratch/kstoreyf/muchisimocks/muchisimocks_lib{tag_mocks}'
    # need to make mock dir now so we can move param files into it
    Path.mkdir(Path(dir_mocks), parents=True, exist_ok=True)
    print(f"Made mock dir {dir_mocks}", flush=True)
    
    param_dict = utils.cosmo_dict_quijote
    seed = idx_mock
    print("param_dict:", param_dict, flush=True)
    
    run_single(param_dict, seed, idx_mock, dir_mocks,
               n_grid=n_grid, n_grid_target=n_grid_target, box_size=box_size,
               n_threads_bacco=n_threads_bacco, n_workers_m2m=n_workers_m2m,
               subdir_prefix='mock')


def run_single(param_dict, seed, idx_mock, dir_mocks,
               n_grid=512, n_grid_target=128, box_size=1000.,
               n_threads_bacco=8, n_workers_m2m=0,
               subdir_prefix='LH'):
    
    deconvolve_lr_field = True
    run_zspace = True
    
    save_intermeds = False
    save_hr_field = False
    
    overwrite_m2m_disp = False
    overwrite_m2m_vel = False
    overwrite_bfields = False
    
    bacco.configuration.update({'pk':{'boltzmann_solver': 'CLASS'}})
    bacco.configuration.update({'pknbody' : {'ngrid'  :  n_grid}})
    bacco.configuration.update({'scaling' : {'disp_ngrid' : n_grid}})
    bacco.configuration.update({'number_of_threads': n_threads_bacco})
    
    #previously was '.'; only works if running script from m2m dir
    #dir_m2m = '/dipc/kstoreyf/external/map2map_emu'
    dir_m2m = '/home/kstoreyf/external/map2map_emu'
    
    #### Start running for particular LH
    
    # these are the paths for the final data products, if exist 
    # and not overwrite, exit now! if want high res / not kcut, need to modify
    dir_LH = f'{dir_mocks}/{subdir_prefix}{idx_mock}'
    fn_bfields_kcut_deconvolved = f'{dir_LH}/bias_fields_eul_deconvolved_{idx_mock}.npy'
    tag_zspace = '_zspace'
    fn_bfields_zspace_kcut_deconvolved = f'{dir_LH}/bias_fields_eul{tag_zspace}_deconvolved_{idx_mock}.npy'

    if run_zspace:
        if os.path.exists(fn_bfields_kcut_deconvolved) and \
           os.path.exists(fn_bfields_zspace_kcut_deconvolved) and \
           not overwrite_bfields:
            print(f"Real-space and z-space bias fields (deconvolved, kcut) for LH {idx_mock} already exists and overwrite_bfields = {overwrite_bfields}, our work here is done!", flush=True)
            # NOTE this must be return and not exit to work in a loop
            return
    else:
        if os.path.exists(fn_bfields_kcut_deconvolved) and not overwrite_bfields:
            print(f"Bias field (deconvolved, kcut) for LH {idx_mock} already exists and overwrite_bfields = {overwrite_bfields}, our work here is done!", flush=True)
            return
        
    print(f"Starting LH {idx_mock}", flush=True)
    fn_disp = f'{dir_LH}/pred_disp.npy'
    fn_vel = f'{dir_LH}/pred_vel.npy'

    ### ADDING FOR NOW AS FIX
    # if not (os.path.exists(fn_disp) or os.path.exists(fn_vel)):
    #    print("Don't have the disp and vel fields for this LH, skipping", flush=True)
    #    return 
    
    start = time.time()
    timenow = start
    
    print("dir_LH:", dir_LH, flush=True)
    Path.mkdir(Path(dir_LH), parents=True, exist_ok=True)

    ## Start cosmology class
    expfactor = 1.0
    cosmo = utils.get_cosmo(param_dict, a_scale=expfactor, sim_name='quijote')
    print(cosmo.pars, flush=True)

    # We also need the parameters in this order in a text file for m2m
    #pars_arr = np.array([Omega0, OmegaBaryon, HubbleParam, ns, sigma8])
    param_names_m2m_ordered = ['omega_cold', 'omega_baryon', 'hubble', 'ns', 'sigma8_cold']
    pars_arr = np.array([param_dict[pn] for pn in param_names_m2m_ordered])
    fn_params_m2m = f'{dir_LH}/cosmo_pars_m2m.txt'
    np.savetxt(fn_params_m2m, pars_arr.T)

    # CREATE A ZA SIMULATION
    fn_ZA_disp = f'{dir_LH}/ZA_disp.npy'
    fn_lin = f'{dir_LH}/lin_field.npy'
    fn_ZA_vel = f'{dir_LH}/ZA_vel.npy'
    
    # need the sim object later on, so for now let's recreate even if 
    # we have the ZA fields saved
    # if overwrite_ZA_fields \
    #     or not os.path.exists(fn_ZA_disp) \
    #     or not os.path.exists(fn_lin) \
    #     or (run_zspace and not os.path.exists(fn_ZA_vel)):
    print("Generating ZA sim", flush=True)
    sim, disp_field = bacco.utils.create_lpt_simulation(cosmo, box_size, Nmesh=n_grid, Seed=seed,
                                                        FixedInitialAmplitude=False, InitialPhase=0, 
                                                        expfactor=expfactor, LPT_order=1, order_by_order=None,
                                                        phase_type=1, ngenic_phases=True, return_disp=True, 
                                                        sphere_mode=0)
    timeprev = timenow
    timenow = time.time()
    print(f"time to create lpt sim: {timenow-timeprev:.2f} s ({(timenow-timeprev)/60:.2f} min)", flush=True)

    print("Saving sims and params", flush=True)

    np.save(fn_ZA_disp, disp_field, allow_pickle=True)
    norm=n_grid**3.
    np.save(fn_lin, sim.linear_field[0]*norm, allow_pickle=True)
    if run_zspace:
        #np.save(fn_ZA_vel, sim.sdm['vel'].reshape((-1,n_grid,n_grid,n_grid)), allow_pickle=True)
        # Changed the above line to the following, which corrected the dimension ordering of vel field
        velocities = sim.sdm['vel']
        vel_field = velocities.reshape((n_grid,n_grid,n_grid,-1))
        vel_field = vel_field.transpose(3,0,1,2)
        np.save(fn_ZA_vel, vel_field, allow_pickle=True)

    # RUN MAP2MAP
    if overwrite_m2m_disp or not os.path.exists(fn_disp):
        print("Running map2map displacements", flush=True)
        ## Positions
        # if i don't pass num-threads, it tries to read from slurm, but i'm not running w slurm
        # num-threads only for when on CPU, not if we have cuda/GPU
        os.system(f'python {dir_m2m}/m2m.py test --num-threads 1 ' 
                f'--test-in-patterns "{fn_ZA_disp}" ' 
                f'--test-tgt-patterns "{fn_ZA_disp}" '
                '--in-norms "cosmology.dis" --tgt-norms "cosmology.dis" '
                '--crop 128 --crop-step 128 --pad 48 '
                f'--model d2d.StyledVNet --batches 1 --loader-workers {n_workers_m2m} '
                f'--load-state "{dir_m2m}/map2map/weights/d2d_weights.pt" '
                f'--callback-at "{dir_m2m}" ' 
                f'--test-style-pattern "{dir_LH}/cosmo_pars_m2m.txt"')
        # hacked m2m to get this to work - takes output name from --test-tgt-patterns,
        # so it's already in proper directory; m2m adds "_out", so it's a diff name;
        # copied from fields.py in map2map
        print("Ran map2map disp")
        fn_disp_m2m = '_out'.join(os.path.splitext(fn_ZA_disp)) 
        os.system(f'mv {fn_disp_m2m} {fn_disp}')
        
        timeprev = timenow
        timenow = time.time()
        print(f"TIME for running map2map positions: {timenow-timeprev:.2f} s ({(timenow-timeprev)/60:.2f} min)", flush=True)

    if overwrite_m2m_vel or not os.path.exists(fn_vel):

        ## Velocities (for RSD)
        if run_zspace:
            print("Running map2map velocities")
            os.system(f'python {dir_m2m}/m2m.py test --num-threads 1 ' 
                f'--test-in-patterns "{fn_ZA_vel}" ' 
                f'--test-tgt-patterns "{fn_ZA_vel}" '
                '--in-norms "cosmology.vel" --tgt-norms "cosmology.vel" '
                '--crop 128 --crop-step 128 --pad 48 '
                f'--model d2d.StyledVNet --batches 1 --loader-workers {n_workers_m2m} '
                f'--load-state "{dir_m2m}/map2map/weights/v2halov_weights.pt" '
                f'--callback-at "{dir_m2m}" ' 
                f'--test-style-pattern "{dir_LH}/cosmo_pars_m2m.txt"')
            # hacked m2m to get this to work - takes output name from --test-tgt-patterns,
            # so it's already in proper directory; m2m adds "_out", so it's a diff name;
            # copied from fields.py in map2map
            fn_vel_m2m = '_out'.join(os.path.splitext(fn_ZA_vel)) 
            os.system(f'mv {fn_vel_m2m} {fn_vel}')

            timeprev = timenow
            timenow = time.time()
            print(f"TIME for running map2map velocities: {timenow-timeprev:.2f} s ({(timenow-timeprev)/60:.2f} min)", flush=True)
                        
    # COMPUTE BIAS MODEL
    print("Reloading map2map result", flush=True)
    ## Read displacement, velocities and linear density
    pred_disp = np.load(fn_disp)
    dens_lin = np.load(f'{dir_LH}/lin_field.npy')

    ## Create regular grid and displace particles
    print("Generating grid", flush=True)
    grid = bacco.visualization.uniform_grid(npix=n_grid, L=box_size, ndim=3, bounds=False)

    print("Adding predicted displacements", flush=True)
    pred_pos = bacco.scaler.add_displacement(None,
                                    pred_disp,
                                    box=box_size,
                                    pos=grid.reshape(-1,3),
                                    vel=None,
                                    vel_factor=0,
                                    verbose=True)[0]
    timeprev = timenow
    timenow = time.time()
    print(f"TIME for adding displacements: {timenow-timeprev:.2f} s ({(timenow-timeprev)/60:.2f} min)", flush=True)

    print("Generating bias fields from particle positions", flush=True)
    
    # Choose 0.75 as damping scale to match emulator
    damping_scale = 0.75
    predicted_positions_to_bias_fields(n_grid, n_grid_target, box_size, sim, 
                                        dens_lin, pred_pos, damping_scale,
                                        save_hr_field, deconvolve_lr_field, save_intermeds,
                                        dir_LH, idx_mock, tag_bfields='')

    ## Include RSD
    if run_zspace:
        velocities = fv2bro(np.load(fn_vel)).copy(order='C')
        pred_pos_zspace = bacco.statistics.compute_zsd(pred_pos, velocities, 
                                                        cosmo, box_size, zspace_axis=2)
        predicted_positions_to_bias_fields(n_grid, n_grid_target, box_size, sim, 
                                        dens_lin, pred_pos_zspace, damping_scale,
                                        save_hr_field, deconvolve_lr_field, save_intermeds,
                                        dir_LH, idx_mock, tag_bfields=tag_zspace)
        
    # clean up intermediate data products
    if not save_intermeds:
        fns_to_remove = [fn_disp, fn_vel, fn_ZA_disp, fn_ZA_vel, fn_lin]
        for fn_to_remove in fns_to_remove:
            if os.path.isfile(fn_to_remove):
                os.system(f'rm {fn_to_remove}')   
        
    timenow = time.time()
    print(f"TIME TOTAL for LH {idx_mock}: {timenow-start} s", flush=True)
    return



def run(idx_mock):
    
    print("Setting up", flush=True)
    n_grid = 512
    n_grid_target = 128

    #tag_params = f'_p3_n50'
    #tag_params = f'_p3_n10'
    tag_params = f'_p5_n10000'
    tag_mocks = f'{tag_params}'
    #tag_mocks = f'{tag_params}_timetests'
    box_size = 1000.

    # uses 12 GB, indep of number of threads
    # 1 job: ~200s = ~3 min each for pos / vel
    # 2 jobs: ~400s = ~7 min each for pos / vel
    # 3 jobs: ~400-500s = ~8 min each for pos / vel
    n_workers_m2m = 0 #0 disables multiprocessing
    n_threads_bacco = 8
    print(f"n_workers_m2m = {n_workers_m2m}, n_threads_bacco = {n_threads_bacco}", flush=True)
    deconvolve_lr_field = True
    run_zspace = True
    
    save_intermeds = False
    save_hr_field = False
    
    overwrite_m2m_disp = False
    overwrite_m2m_vel = False
    overwrite_bfields = False
    
    #previously was '.'; only works if running script from m2m dir
    #dir_m2m = '/dipc/kstoreyf/external/map2map_emu'
    dir_m2m = '/home/kstoreyf/external/map2map_emu'
    #dir_mocks = f'/dipc/kstoreyf/muchisimocks/data/cosmolib{tag_mocks}'
    #dir_mocks = f'/cosmos_storage/cosmosims/muchisimocks/muchisimocks_lib{tag_mocks}'
    dir_mocks = f'/scratch/kstoreyf/muchisimocks/muchisimocks_lib{tag_mocks}'
    # need to make mock dir now so we can move param files into it
    Path.mkdir(Path(dir_mocks), parents=True, exist_ok=True)
    print(f"Made mock dir {dir_mocks}", flush=True)
    
    # Deal with cosmological parameters
    
    dir_params = '../data/params'
    fn_params_orig = f'{dir_params}/params_lh{tag_params}.txt'
    fn_params_fixed_orig = f'{dir_params}/params_fixed{tag_params}.txt'
    # copy parameters to mocks folder to make sure we know which params were used
    fn_params = f'{dir_mocks}/params_lh{tag_params}.txt'
    fn_params_fixed = f'{dir_mocks}/params_fixed{tag_params}.txt'
    if not os.path.exists(fn_params):
        os.system(f'cp {fn_params_orig} {fn_params}')
    if not os.path.exists(fn_params_fixed):
        os.system(f'cp {fn_params_fixed_orig} {fn_params_fixed}')
    params_df = pd.read_csv(fn_params, index_col=0)
    param_dict_fixed = pd.read_csv(fn_params_fixed).loc[0].to_dict()

    bacco.configuration.update({'pk':{'boltzmann_solver': 'CLASS'}})
    bacco.configuration.update({'pknbody' : {'ngrid'  :  n_grid}})
    bacco.configuration.update({'scaling' : {'disp_ngrid' : n_grid}})
    bacco.configuration.update({'number_of_threads': n_threads_bacco})

    
    #### Start running for particular LH
    
    # these are the paths for the final data products, if exist 
    # and not overwrite, exit now! if want high res / not kcut, need to modify
    dir_LH = f'{dir_mocks}/LH{idx_mock}'
    fn_bfields_kcut_deconvolved = f'{dir_LH}/bias_fields_eul_deconvolved_{idx_mock}.npy'
    tag_zspace = '_zspace'
    fn_bfields_zspace_kcut_deconvolved = f'{dir_LH}/bias_fields_eul{tag_zspace}_deconvolved_{idx_mock}.npy'

    if run_zspace:
        if os.path.exists(fn_bfields_kcut_deconvolved) and \
           os.path.exists(fn_bfields_zspace_kcut_deconvolved) and \
           not overwrite_bfields:
            print(f"Real-space and z-space bias fields (deconvolved, kcut) for LH {idx_mock} already exists and overwrite_bfields = {overwrite_bfields}, our work here is done!", flush=True)
            # NOTE this must be return and not exit to work in a loop
            return
    else:
        if os.path.exists(fn_bfields_kcut_deconvolved) and not overwrite_bfields:
            print(f"Bias field (deconvolved, kcut) for LH {idx_mock} already exists and overwrite_bfields = {overwrite_bfields}, our work here is done!", flush=True)
            return
        
    print(f"Starting LH {idx_mock}", flush=True)
    fn_disp = f'{dir_LH}/pred_disp.npy'
    fn_vel = f'{dir_LH}/pred_vel.npy'

    ### ADDING FOR NOW AS FIX
    # if not (os.path.exists(fn_disp) or os.path.exists(fn_vel)):
    #    print("Don't have the disp and vel fields for this LH, skipping", flush=True)
    #    return 
    
    start = time.time()
    timenow = start
    
    print("dir_LH:", dir_LH, flush=True)
    Path.mkdir(Path(dir_LH), parents=True, exist_ok=True)
    
    param_dict = params_df.loc[idx_mock].to_dict()
    param_dict.update(param_dict_fixed)
    seed = idx_mock
    print("param_dict:", param_dict, flush=True)

    ## Start cosmology class
    expfactor = 1.0
    cosmo = utils.get_cosmo(param_dict, a_scale=expfactor, sim_name='quijote')
    print(cosmo.pars, flush=True)

    # We also need the parameters in this order in a text file for m2m
    #pars_arr = np.array([Omega0, OmegaBaryon, HubbleParam, ns, sigma8])
    param_names_m2m_ordered = ['omega_cold', 'omega_baryon', 'hubble', 'ns', 'sigma8_cold']
    pars_arr = np.array([param_dict[pn] for pn in param_names_m2m_ordered])
    fn_params_m2m = f'{dir_LH}/cosmo_pars_m2m.txt'
    np.savetxt(fn_params_m2m, pars_arr.T)

    # CREATE A ZA SIMULATION
    fn_ZA_disp = f'{dir_LH}/ZA_disp.npy'
    fn_lin = f'{dir_LH}/lin_field.npy'
    fn_ZA_vel = f'{dir_LH}/ZA_vel.npy'
    
    # need the sim object later on, so for now let's recreate even if 
    # we have the ZA fields saved
    # if overwrite_ZA_fields \
    #     or not os.path.exists(fn_ZA_disp) \
    #     or not os.path.exists(fn_lin) \
    #     or (run_zspace and not os.path.exists(fn_ZA_vel)):
    print("Generating ZA sim", flush=True)
    sim, disp_field = bacco.utils.create_lpt_simulation(cosmo, box_size, Nmesh=n_grid, Seed=seed,
                                                        FixedInitialAmplitude=False, InitialPhase=0, 
                                                        expfactor=expfactor, LPT_order=1, order_by_order=None,
                                                        phase_type=1, ngenic_phases=True, return_disp=True, 
                                                        sphere_mode=0)
    timeprev = timenow
    timenow = time.time()
    print(f"time to create lpt sim: {timenow-timeprev:.2f} s ({(timenow-timeprev)/60:.2f} min)", flush=True)

    print("Saving sims and params", flush=True)

    np.save(fn_ZA_disp, disp_field, allow_pickle=True)
    norm=n_grid**3.
    np.save(fn_lin, sim.linear_field[0]*norm, allow_pickle=True)
    if run_zspace:
        #np.save(fn_ZA_vel, sim.sdm['vel'].reshape((-1,n_grid,n_grid,n_grid)), allow_pickle=True)
        # Changed the above line to the following, which corrected the dimension ordering of vel field
        velocities = sim.sdm['vel']
        vel_field = velocities.reshape((n_grid,n_grid,n_grid,-1))
        vel_field = vel_field.transpose(3,0,1,2)
        np.save(fn_ZA_vel, vel_field, allow_pickle=True)

    # RUN MAP2MAP
    if overwrite_m2m_disp or not os.path.exists(fn_disp):
        print("Running map2map", flush=True)
        ## Positions
        # if i don't pass num-threads, it tries to read from slurm, but i'm not running w slurm
        # num-threads only for when on CPU, not if we have cuda/GPU
        os.system(f'python {dir_m2m}/m2m.py test --num-threads 1 ' 
                f'--test-in-patterns "{fn_ZA_disp}" ' 
                f'--test-tgt-patterns "{fn_ZA_disp}" '
                '--in-norms "cosmology.dis" --tgt-norms "cosmology.dis" '
                '--crop 128 --crop-step 128 --pad 48 '
                f'--model d2d.StyledVNet --batches 1 --loader-workers {n_workers_m2m} '
                f'--load-state "{dir_m2m}/map2map/weights/d2d_weights.pt" '
                f'--callback-at "{dir_m2m}" ' 
                f'--test-style-pattern "{dir_LH}/cosmo_pars_m2m.txt"')
        # hacked m2m to get this to work - takes output name from --test-tgt-patterns,
        # so it's already in proper directory; m2m adds "_out", so it's a diff name;
        # copied from fields.py in map2map
        fn_disp_m2m = '_out'.join(os.path.splitext(fn_ZA_disp)) 
        os.system(f'mv {fn_disp_m2m} {fn_disp}')
        
        timeprev = timenow
        timenow = time.time()
        print(f"TIME for running map2map positions: {timenow-timeprev:.2f} s ({(timenow-timeprev)/60:.2f} min)", flush=True)

    if overwrite_m2m_vel or not os.path.exists(fn_vel):

        ## Velocities (for RSD)
        if run_zspace:
            os.system(f'python {dir_m2m}/m2m.py test --num-threads 1 ' 
                f'--test-in-patterns "{fn_ZA_vel}" ' 
                f'--test-tgt-patterns "{fn_ZA_vel}" '
                '--in-norms "cosmology.vel" --tgt-norms "cosmology.vel" '
                '--crop 128 --crop-step 128 --pad 48 '
                f'--model d2d.StyledVNet --batches 1 --loader-workers {n_workers_m2m} '
                f'--load-state "{dir_m2m}/map2map/weights/v2halov_weights.pt" '
                f'--callback-at "{dir_m2m}" ' 
                f'--test-style-pattern "{dir_LH}/cosmo_pars_m2m.txt"')
            # hacked m2m to get this to work - takes output name from --test-tgt-patterns,
            # so it's already in proper directory; m2m adds "_out", so it's a diff name;
            # copied from fields.py in map2map
            fn_vel_m2m = '_out'.join(os.path.splitext(fn_ZA_vel)) 
            os.system(f'mv {fn_vel_m2m} {fn_vel}')

            timeprev = timenow
            timenow = time.time()
            print(f"TIME for running map2map velocities: {timenow-timeprev:.2f} s ({(timenow-timeprev)/60:.2f} min)", flush=True)
                        
    # COMPUTE BIAS MODEL
    print("Reloading map2map result", flush=True)
    ## Read displacement, velocities and linear density
    pred_disp = np.load(fn_disp)
    dens_lin = np.load(f'{dir_LH}/lin_field.npy')

    ## Create regular grid and displace particles
    print("Generating grid", flush=True)
    grid = bacco.visualization.uniform_grid(npix=n_grid, L=box_size, ndim=3, bounds=False)

    print("Adding predicted displacements", flush=True)
    pred_pos = bacco.scaler.add_displacement(None,
                                    pred_disp,
                                    box=box_size,
                                    pos=grid.reshape(-1,3),
                                    vel=None,
                                    vel_factor=0,
                                    verbose=True)[0]
    timeprev = timenow
    timenow = time.time()
    print(f"TIME for adding displacements: {timenow-timeprev:.2f} s ({(timenow-timeprev)/60:.2f} min)", flush=True)

    print("Generating bias fields from particle positions", flush=True)
    
    # Choose 0.75 as damping scale to match emulator
    damping_scale = 0.75
    predicted_positions_to_bias_fields(n_grid, n_grid_target, box_size, sim, 
                                        dens_lin, pred_pos, damping_scale,
                                        save_hr_field, deconvolve_lr_field, save_intermeds,
                                        dir_LH, idx_mock, tag_bfields='')

    ## Include RSD
    if run_zspace:
        velocities = fv2bro(np.load(fn_vel)).copy(order='C')
        pred_pos_zspace = bacco.statistics.compute_zsd(pred_pos, velocities, 
                                                        cosmo, box_size, zspace_axis=2)
        predicted_positions_to_bias_fields(n_grid, n_grid_target, box_size, sim, 
                                        dens_lin, pred_pos_zspace, damping_scale,
                                        save_hr_field, deconvolve_lr_field, save_intermeds,
                                        dir_LH, idx_mock, tag_bfields=tag_zspace)
        
    # clean up intermediate data products
    if not save_intermeds:
        fns_to_remove = [fn_disp, fn_vel, fn_ZA_disp, fn_ZA_vel, fn_lin]
        for fn_to_remove in fns_to_remove:
            if os.path.isfile(fn_to_remove):
                os.system(f'rm {fn_to_remove}')   
        
    timenow = time.time()
    print(f"TIME TOTAL for LH {idx_mock}: {timenow-start} s", flush=True)
    return

    

def predicted_positions_to_bias_fields(n_grid, n_grid_target, box_size, sim, 
                                        dens_lin, pred_pos, damping_scale,
                                        save_hr_field, deconvolve_lr_field, save_intermeds,
                                        dir_LH, idx_mock, tag_bfields=''):

    timenow = time.time()
    
    ## Start bias model class
    interlacing = False
    print("Setting up bias model", flush=True)
    bmodel = bacco.BiasModel(sim=sim, #linear_delta=dens_lin, 
                            ngrid=n_grid, ngrid1=None, 
                            sdm=False, mode="dm",
                            npart_for_fake_sim=n_grid, damping_scale=damping_scale, 
                            bias_model='expansion', deposit_method="cic", 
                            use_displacement_of_nn=False, interlacing=interlacing, 
                            )

    ## Compute lagrangian fields
    print("Computing lagrangian fields", flush=True)
    bias_fields = bmodel.bias_terms_lag()

    ## Compute eulerian fields
    print("Computing eulerian fields", flush=True)
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
        print(f"Saving full eulerian fields {tag_bfields}", flush=True)
        np.save(f'{dir_LH}/bias_fields_eul{tag_bfields}_hr_{idx_mock}.npy', bias_terms_eul_pred, allow_pickle=True)
    timeprev = timenow
    timenow = time.time()
    print(f"TIME for making eulerian fields {tag_bfields}: {timenow-timeprev:.2f} s ({(timenow-timeprev)/60:.2f} min)", flush=True)

    print("Cutting k-modes")
    bias_terms_eul_pred_kcut = remove_lowk_modes(bias_terms_eul_pred, box_size, n_grid_target)
    timeprev = timenow
    timenow = time.time()
    print(f"TIME for cutting fields to certain kmode {tag_bfields}: {timenow-timeprev:.2f} s ({(timenow-timeprev)/60:.2f} min)", flush=True)
    if save_intermeds:
        print(f"Saving k-cut, non-deconvolved eulerian fields {tag_bfields}", flush=True)
        np.save(f'{dir_LH}/bias_fields_eul{tag_bfields}_{idx_mock}.npy', bias_terms_eul_pred_kcut, allow_pickle=True)
    
    # Deconvolve bias fields - deconvolving the field with the low-k modes already removed bc much faster.
    # Showed that it gets essentially same result as deconvolving the HR field and then cutting the low-k-modes
    if deconvolve_lr_field:
        print(f"Deconvolving k-cut bias fields {tag_bfields}", flush=True)
        bias_terms_eul_pred_kcut_deconvolved = deconvolve_bias_field(bias_terms_eul_pred_kcut,
                                                                     n_grid)
        # for some reason this pb function turns our float32 array into float64, 
        # convert back before saving
        bias_terms_eul_pred_kcut_deconvolved = bias_terms_eul_pred_kcut_deconvolved.astype(np.float32)
        
        fn_bfields_kcut_deconvolved = f'{dir_LH}/bias_fields_eul{tag_bfields}_deconvolved_{idx_mock}.npy'
        np.save(fn_bfields_kcut_deconvolved, bias_terms_eul_pred_kcut_deconvolved, allow_pickle=True)
        timeprev = timenow
        timenow = time.time()
        print(f"TIME for deconvolving k-cut fields {tag_bfields}: {timenow-timeprev:.2f} s ({(timenow-timeprev)/60:.2f} min)", flush=True)



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

