import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pyfftw
import time

import bacco

import utils


os.environ["MKL_SERVICE_FORCE_INTEL"] = str(1)


def main():
    ngrid = 512
    ngrid_target = 128

    box_size = 1000.
    n_threads_bacco = 12
    n_threads_m2m = 1

    bacco.configuration.update({'pk':{'boltzmann_solver': 'CLASS'}})
    bacco.configuration.update({'pknbody' : {'ngrid'  :  ngrid}})
    bacco.configuration.update({'scaling' : {'disp_ngrid' : ngrid}})
    bacco.configuration.update({'number_of_threads': n_threads_bacco})

    idx_LH_start = 0
    idx_LH_end = 1

    save_intermeds = True

    tag_lib = '_ksf'
    #tag_lib = '_FixedPk'
    if 'FixedPk' in tag_lib:
        FixedInitialAmplitude = True
    else:
        FixedInitialAmplitude = False

    #previously was '.'; only works if running script from m2m dir
    dir_m2m = '/dipc/kstoreyf/external/map2map_emu'
    dir_lib = f'/dipc/kstoreyf/muchisimocks/data/cosmolib{tag_lib}'
    # for now, copying cosmology from the main cosmolib; TODO change this later!
    dir_cosmopars = '/dipc/kstoreyf/muchisimocks/data/cosmolib'    
    
    for idx_LH in range(idx_LH_start, idx_LH_end):
        
        start = time.time()
        
        dir_lh = f'{dir_lib}/LH{idx_LH}'
        print("dir_lh:", dir_lh)
        Path.mkdir(Path(dir_lh), parents=True, exist_ok=True)
        if save_intermeds:
            dir_save = dir_lh
        else:
            dir_save = dir_m2m
            
        # get cosmology
        fn_cosmopars_orig = f'{dir_cosmopars}/LH{idx_LH}/cosmo_{idx_LH}.txt'
        fn_cosmopars = f'{dir_lh}/cosmo_{idx_LH}.txt'
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
        np.savetxt(f'{dir_save}/cosmo_pars_m2m.txt', pars_arr.T)


        # CREATE A ZA SIMULATION
        print("Generating ZA sim")
        sim, disp_field = bacco.utils.create_lpt_simulation(cosmo, box_size, Nmesh=ngrid, Seed=Seed,
                                                            FixedInitialAmplitude=FixedInitialAmplitude,InitialPhase=0, 
                                                            expfactor=1, LPT_order=1, order_by_order=None,
                                                            phase_type=1, ngenic_phases=True, return_disp=True, 
                                                            sphere_mode=0)
        print(f"time after creating lpt sim: {time.time()-start} s")

        print("Saving sims and params")
        np.save(f'{dir_save}/ZA_disp.npy', disp_field,allow_pickle=True)
        norm=ngrid**3.
        np.save(f'{dir_save}/lin_field.npy', sim.linear_field[0]*norm,allow_pickle=True)
        #np.save('ZA_vel.npy', sim.sdm['vel'].reshape((3,512,512,512)), allow_pickle=True)

        # RUN MAP2MAP
        print("Running map2map")
        ## Positions
        # if i don't pass num-threads, it tries to read from slurm, but i'm not running w slurm
        
        # os.system('python m2m.py test --test-in-patterns "ZA_disp.npy" --test-tgt-patterns "ZA_disp.npy" --in-norms "cosmology.dis" --tgt-norms "cosmology.dis" --crop 128 --crop-step 128 --pad 48 --model d2d.StyledVNet --batches 1 --loader-workers 7 --load-state "map2map/weights/d2d_weights.pt" --callback-at "." --test-style-pattern "cosmo_pars.txt"')
        os.system(f'python {dir_m2m}/m2m.py test --num-threads {n_threads_m2m} ' 
                  f'--test-in-patterns "{dir_save}/ZA_disp.npy" ' 
                  f'--test-tgt-patterns "{dir_save}/ZA_disp.npy" '
                  '--in-norms "cosmology.dis" --tgt-norms "cosmology.dis" '
                  '--crop 128 --crop-step 128 --pad 48 '
                  '--model d2d.StyledVNet --batches 1 --loader-workers 7 '
                  f'--load-state "{dir_m2m}/map2map/weights/d2d_weights.pt" '
                  f'--callback-at "{dir_m2m}" ' 
                  f'--test-style-pattern "{dir_save}/cosmo_pars_m2m.txt"')

        print("Renaming map2map result")        
        fn_disp = f'{dir_save}/pred_disp.npy'
        os.system(f'mv ._out.npy {fn_disp}')
        print(f"time after running map2map: {time.time()-start} s")

        ## Velocities
        #os.system('python m2m.py test --test-in-patterns "ZA_vel.npy" --test-tgt-patterns "ZA_vel.npy" --in-norms "cosmology.vel" --tgt-norms "cosmology.vel" --crop 128 --crop-step 128 --pad 48 --model d2d.StyledVNet --batches 1 --loader-workers 7 --load-state "v2halov_weights.pt" --callback-at "." --test-style-pattern "cosmo_pars.txt"')
        #os.system('mv ._out.npy pred_vel.npy')


        # COMPUTE BIAS MODEL

        print("Reloading map2map result")
        ## Read displacement, velocities and linear density
        pred_disp = np.load(f'{dir_save}/pred_disp.npy')
        #velocities = fv2bro(np.load('pred_vel.npy')).copy(order='C')
        dens_lin = np.load(f'{dir_save}/lin_field.npy')

        ## Create regular grid and displace particles
        print("Generating grid")
        grid = bacco.visualization.uniform_grid(npix=ngrid, L=box_size, ndim=3, bounds=False)

        print("Adding predicted displacements")
        pred_pos = bacco.scaler.add_displacement(None,
                                        pred_disp,
                                        box=box_size,
                                        pos=grid.reshape(-1,3),
                                        vel=None,
                                        vel_factor=0,
                                        verbose=True)[0]
        print(f"time after adding displacements: {time.time()-start} s")


        ## Include RSD

        #pred_pos = bacco.statistics.compute_zsd(pred_pos, velocities, cosmo,box_size, zspace_axis=2)

        ## Start bias model class

        #k_nyq = np.pi * ngrid / box_size
        damping_scale = 0.7 #k_nyq
        interlacing = False

        print("Setting up bias model")
        bmodel = bacco.BiasModel(sim=sim, linear_delta=dens_lin, ngrid=ngrid, ngrid1=None, 
                                sdm=False, mode="dm",
                                npart_for_fake_sim=ngrid, damping_scale=damping_scale, 
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
            bias_terms_pred = bacco.statistics.compute_mesh(ngrid=ngrid, box=box_size, pos=pred_pos, 
                                    mass = (bias_fields[ii]).flatten(), deposit_method='cic', 
                                    interlacing=interlacing)
            bias_terms_eul_pred.append(bias_terms_pred)
        bias_terms_eul_pred = np.array(bias_terms_eul_pred)
        
        print("Saving full eulerian fields")
        #np.save(outpath+'/Eulerian_fields_lr_'+str(cind)+'_full.npy', bias_terms_eul_pred, allow_pickle=True)
        np.save(f'{dir_lh}/Eulerian_fields_hr_{idx_LH}.npy', bias_terms_eul_pred, allow_pickle=True)
        print(f"time after making eulerian fields: {time.time()-start} s")

        print("Cutting k-modes")
        bias_terms_eul_pred_kcut = downsample_field(bias_terms_eul_pred, box_size, ngrid_target)

        ## Save eulerian fields
        print("Saving cut eulerian fields")
        np.save(f'{dir_lh}/Eulerian_fields_lr_{idx_LH}.npy', bias_terms_eul_pred_kcut, allow_pickle=True)
        print(f"time after cutting fields to certain kmode: {time.time()-start} s")

        print(f"time for single LH: {time.time()-start} s")


def downsample_field(bias_terms_eul_pred, box_size, ngrid_target):
    ngrid = bias_terms_eul_pred.shape[-1]
    k_nyq = np.pi/box_size*ngrid_target
    kmesh = bacco.visualization.np_get_kmesh( (ngrid, ngrid, ngrid), box_size, real=True)
    mask = (kmesh[:,:,:,0]<=k_nyq) & (kmesh[:,:,:,1]<=k_nyq) & (kmesh[:,:,:,2]<=k_nyq) & (kmesh[:,:,:,0]>-k_nyq) & (kmesh[:,:,:,1]>-k_nyq) & (kmesh[:,:,:,2]>-k_nyq)
    bias_terms_eul_pred_kcut=[]
    for fid in range(5):
        field = bias_terms_eul_pred[fid][0]
        deltak = pyfftw.builders.rfftn(field, auto_align_input=False, auto_contiguous=False, avoid_copy=True)
        deltakcut = deltak()[mask]
        deltakcut= deltakcut.reshape(ngrid_target, ngrid_target, ngrid_target/2+1)
        delta = pyfftw.builders.irfftn(deltakcut, axes=(0,1,2))()
        bias_terms_eul_pred_kcut.append(delta)
    bias_terms_eul_pred_kcut = np.array(bias_terms_eul_pred_kcut)
    return bias_terms_eul_pred_kcut


def fv2bro(t_fv_field) :
    '''Returns back row ordered array (shape ngrid * ngrid * ngrid, 3) from front vector (shape 3, ngrid, ngrid, ngrid)'''
    return np.reshape(t_fv_field, (3, int(t_fv_field.size / 3))).T



if __name__=='__main__':
    main()