import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import bacco

os.environ["MKL_SERVICE_FORCE_INTEL"] = str(1)


def fv2bro(t_fv_field) :
    '''Returns back row ordered array (shape ngrid * ngrid * ngrid, 3) from front vector (shape 3, ngrid, ngrid, ngrid)'''
    return np.reshape(t_fv_field, (3, int(t_fv_field.size / 3))).T

npart = 512
boxsize = 1000.
n_threads = 12

bacco.configuration.update({'pk':{'boltzmann_solver': 'CLASS'}})
bacco.configuration.update({'pknbody' : {'ngrid'  :  npart}})
bacco.configuration.update({'scaling' : {'disp_ngrid' : npart}})
bacco.configuration.update({'number_of_threads': n_threads})

start_cosmo = 0
end_cosmo = 1

tag_m2m = '_FixedPk'
if 'FixedPk' in tag_m2m:
    FixedInitialAmplitude = True
else:
    FixedInitialAmplitude = False
    
for cind in range(start_cosmo,end_cosmo):
    inpath = '/dipc/kstoreyf/muchisimocks/data/cosmolib/'
    cospars = np.loadtxt(inpath+'LH'+str(cind)+'/cosmo_'+str(cind)+'.txt')

    # CHOOSE COSMOLOGY
    Omega0 = cospars[0] #0.3175
    sigma8 = cospars[1] #0.834
    HubbleParam = cospars[2] #0.6711
    OmegaBaryon = cospars[3] #0.049
    ns = cospars[4] #0.9624
    Seed= int(cospars[5]) #1915
    
    print('-------> Seed for the simulation:', Seed)
    
    expfactor = 1.

    pars_arr = np.array([Omega0, OmegaBaryon, HubbleParam, ns, sigma8])

    ## Start cosmology class

    pars = {'omega_cdm':Omega0-OmegaBaryon, 'omega_baryon':OmegaBaryon, 'hubble':HubbleParam, 
            'neutrino_mass':0.0, 'sigma8':sigma8, 'ns':ns, 'expfactor':expfactor}
    cosmo = bacco.Cosmology(**pars) #bacco.Cosmology(**bacco.cosmo_parameters.Planck13)  
    print(cosmo)


    # CREATE A ZA SIMULATION
    print("Generating ZA sim")
    sim, disp_field = bacco.utils.create_lpt_simulation(cosmo, boxsize, Nmesh=npart, Seed=Seed,
                                                        FixedInitialAmplitude=FixedInitialAmplitude,InitialPhase=0, 
                                                        expfactor=1, LPT_order=1, order_by_order=None,
                                                        phase_type=1, ngenic_phases=True, return_disp=True, 
                                                        sphere_mode=0)
    
    # norm=npart**3.

    # print("Saving sims and params")
    # np.save('ZA_disp.npy', disp_field,allow_pickle=True)
    # np.save('lin_field.npy', sim.linear_field[0]*norm,allow_pickle=True)
    # #np.save('ZA_vel.npy', sim.sdm['vel'].reshape((3,512,512,512)), allow_pickle=True)
    # np.savetxt('cosmo_pars.txt', pars_arr.T)

    # # RUN MAP2MAP
    # print("Running map2map")
    # ## Positions
    # # if i don't pass num-threads, it tries to read from slurm, but i'm not running w slurm
    
    # # os.system('python m2m.py test --test-in-patterns "ZA_disp.npy" --test-tgt-patterns "ZA_disp.npy" --in-norms "cosmology.dis" --tgt-norms "cosmology.dis" --crop 128 --crop-step 128 --pad 48 --model d2d.StyledVNet --batches 1 --loader-workers 7 --load-state "map2map/weights/d2d_weights.pt" --callback-at "." --test-style-pattern "cosmo_pars.txt"')
    # os.system('python m2m.py test --num-threads 12 --test-in-patterns "ZA_disp.npy" --test-tgt-patterns "ZA_disp.npy" --in-norms "cosmology.dis" --tgt-norms "cosmology.dis" --crop 128 --crop-step 128 --pad 48 --model d2d.StyledVNet --batches 1 --loader-workers 7 --load-state "map2map/weights/d2d_weights.pt" --callback-at "." --test-style-pattern "cosmo_pars.txt"')

    # print("Renaming map2map result")
    # os.system('mv ._out.npy pred_disp.npy')

    ## Velocities
    #os.system('python m2m.py test --test-in-patterns "ZA_vel.npy" --test-tgt-patterns "ZA_vel.npy" --in-norms "cosmology.vel" --tgt-norms "cosmology.vel" --crop 128 --crop-step 128 --pad 48 --model d2d.StyledVNet --batches 1 --loader-workers 7 --load-state "v2halov_weights.pt" --callback-at "." --test-style-pattern "cosmo_pars.txt"')

    #os.system('mv ._out.npy pred_vel.npy')


    # COMPUTE BIAS MODEL

    print("Reloading map2map result")
    ## Read displacement, velocities and linear density
    pred_disp = np.load('pred_disp.npy')
    #velocities = fv2bro(np.load('pred_vel.npy')).copy(order='C')
    dens_lin = np.load('lin_field.npy')

    ## Create regular grid and displace particles
    print("Generating grid")
    grid = bacco.visualization.uniform_grid(npix=npart, L=boxsize, ndim=3, bounds=False)

    print("Adding predicted displacements")
    pred_pos = bacco.scaler.add_displacement(None,
                                     pred_disp,
                                     box=boxsize,
                                     pos=grid.reshape(-1,3),
                                     vel=None,
                                     vel_factor=0,
                                     verbose=True)[0]


    ## Include RSD

    #pred_pos = bacco.statistics.compute_zsd(pred_pos, velocities, cosmo,boxsize, zspace_axis=2)

    ## Start bias model class

    #k_nyq = np.pi * npart / boxsize
    damping_scale = 0.7 #k_nyq
    interlacing = False

    print("Setting up bias model")
    bmodel = bacco.BiasModel(sim=sim, linear_delta=dens_lin, ngrid=npart, ngrid1=None, 
                             sdm=False, mode="dm",
                             npart_for_fake_sim=npart, damping_scale=damping_scale, 
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
        bias_terms_pred = bacco.statistics.compute_mesh(ngrid=npart, box=boxsize, pos=pred_pos, 
                                  mass = (bias_fields[ii]).flatten(), deposit_method='cic', 
                                  interlacing=interlacing)
        bias_terms_eul_pred.append(bias_terms_pred)
    bias_terms_eul_pred = np.array(bias_terms_eul_pred)
    
    print("Saving full eulerian fields")
    outpath = f'/dipc/kstoreyf/muchisimocks/data/cosmolib{tag_m2m}/LH{str(cind)}'
    Path.mkdir(Path(outpath), parents=True, exist_ok=True)
    np.save(outpath+'/Eulerian_fields_lr_'+str(cind)+'_full.npy', bias_terms_eul_pred, allow_pickle=True)
    
    print("Cutting k-modes")
    from bacco.visualization import np_get_kmesh
    import pyfftw
    ngrid=512
    k_nyq = np.pi/1000*128
    kmesh = np_get_kmesh( (ngrid, ngrid, ngrid), boxsize, real=True)
    mask = (kmesh[:,:,:,0]<=k_nyq) & (kmesh[:,:,:,1]<=k_nyq) & (kmesh[:,:,:,2]<=k_nyq) & (kmesh[:,:,:,0]>-k_nyq) & (kmesh[:,:,:,1]>-k_nyq) & (kmesh[:,:,:,2]>-k_nyq)
    bias_terms_eul_pred_kcut=[]
    for fid in range(5):
        field = bias_terms_eul_pred[fid][0]
        deltak = pyfftw.builders.rfftn(field, auto_align_input=False, auto_contiguous=False, avoid_copy=True)
        deltakcut = deltak()[mask]
        deltakcut= deltakcut.reshape(128, 128, 65)
        delta = pyfftw.builders.irfftn(deltakcut, axes=(0,1,2))()
        bias_terms_eul_pred_kcut.append(delta)
    bias_terms_eul_pred_kcut = np.array(bias_terms_eul_pred_kcut)


    ##### REDUCE DIMENSIONALITY EXAMPLE ####
    #rho = np.random.uniform(0., 1., (32,32,32))
    #print(rho.shape)
    #ns = 2
    #nf = rho.shape[0] // ns
    #rhof = rho.reshape((nf,ns,nf,ns,nf,ns))
    #print(rhof.shape)
    #rholr = np.sum(rhof, axis=(1,3,5))
    #print(rholr.shape)
    #######################################

    ## Save eulerian fields
    print("Saving cut eulerian fields")
    outpath = f'/dipc/kstoreyf/muchisimocks/data/cosmolib{tag_m2m}/LH{str(cind)}'
    print("outpath:", outpath)
    Path.mkdir(Path(outpath), parents=True, exist_ok=True)
    np.save(outpath+'/Eulerian_fields_lr_'+str(cind)+'.npy', bias_terms_eul_pred_kcut, allow_pickle=True)

