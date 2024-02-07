import os
import numpy as np
import matplotlib.pyplot as plt
import bacco

def fv2bro(t_fv_field) :
    '''Returns back row ordered array (shape ngrid * ngrid * ngrid, 3) from front vector (shape 3, ngrid, ngrid, ngrid)'''
    return np.reshape(t_fv_field, (3, int(t_fv_field.size / 3))).T

Seed=1915
npart = 512
boxsize = 1000.

bacco.configuration.update({'pk':{'boltzmann_solver': 'CLASS'}})
bacco.configuration.update({'pknbody' : {'ngrid'  :  npart}})
bacco.configuration.update({'scaling' : {'disp_ngrid' : npart}})

# CHOOSE COSMOLOGY

Omega0 = 0.3175
OmegaBaryon = 0.049
HubbleParam = 0.6711
ns = 0.9624
sigma8 = 0.834
expfactor = 1.

pars_arr = np.array([Omega0, OmegaBaryon, HubbleParam, ns, sigma8])

## Start cosmology class

pars = {'omega_cdm':Omega0-OmegaBaryon, 'omega_baryon':OmegaBaryon, 'hubble':HubbleParam, 
        'neutrino_mass':0.0, 'sigma8':sigma8, 'ns':ns, 'expfactor':expfactor}
cosmo = bacco.Cosmology(**pars) #bacco.Cosmology(**bacco.cosmo_parameters.Planck13)  
print(cosmo)


# CREATE A ZA SIMULATION

sim, disp_field = bacco.utils.create_lpt_simulation(cosmo, boxsize, Nmesh=npart, Seed=Seed,
                                                    FixedInitialAmplitude=False,InitialPhase=0, 
                                                    expfactor=1, LPT_order=1, order_by_order=None,
                                                    phase_type=1, ngenic_phases=True, return_disp=True, 
                                                    sphere_mode=0)

np.save('ZA_disp.npy', disp_field,allow_pickle=True)
np.save('lin_field.npy', sim.linear_field[0],allow_pickle=True)
np.save('ZA_vel.npy', sim.sdm['vel'].reshape((3,512,512,512)), allow_pickle=True)
np.savetxt('cosmo_pars.txt', pars_arr.T)

# RUN MAP2MAP

## Positions
os.system('python m2m.py test --test-in-patterns "ZA_disp.npy" --test-tgt-patterns "ZA_disp.npy" --in-norms "cosmology.dis" --tgt-norms "cosmology.dis" --crop 128 --crop-step 128 --pad 48 --model d2d.StyledVNet --batches 1 --loader-workers 7 --load-state "map2map/weights/d2d_weights.pt" --callback-at "." --test-style-pattern "cosmo_pars.txt"')

os.system('mv ._out.npy pred_disp.npy')

## Velocities
os.system('python m2m.py test --test-in-patterns "ZA_vel.npy" --test-tgt-patterns "ZA_vel.npy" --in-norms "cosmology.vel" --tgt-norms "cosmology.vel" --crop 128 --crop-step 128 --pad 48 --model d2d.StyledVNet --batches 1 --loader-workers 7 --load-state "v2halov_weights.pt" --callback-at "." --test-style-pattern "cosmo_pars.txt"')

os.system('mv ._out.npy pred_vel.npy')


# COMPUTE BIAS MODEL


## Read displacement, velocities and linear density
pred_disp = np.load('pred_disp.npy')
velocities = fv2bro(np.load('pred_vel.npy')).copy(order='C')
dens_lin = np.load('lin_field.npy')

## Create regular grid and displace particles
grid = bacco.visualization.uniform_grid(npix=npart, L=boxsize, ndim=3, bounds=False)

pred_pos = bacco.scaler.add_displacement(None,
                                 pred_disp,
                                 box=boxsize,
                                 pos=grid.reshape(-1,3),
                                 vel=None,
                                 vel_factor=0,
                                 verbose=True)[0]


## Include RSD

pred_pos = bacco.statistics.compute_zsd(pred_pos, velocities, cosmo,boxsize, zspace_axis=2)

## Start bias model class

k_nyq = np.pi * npart / boxsize
damping_scale = k_nyq

bmodel = bacco.BiasModel(sim=sim, linear_delta=dens_lin, ngrid=npart, ngrid1=None, 
                         sdm=False, mode="dm",
                         npart_for_fake_sim=npart, damping_scale=damping_scale, 
                         bias_model='expansion', deposit_method="cic", 
                         use_displacement_of_nn=False, interlacing=False, 
                         )

## Compute lagrangian fields

bias_fields = bmodel.bias_terms_lag()

## Compute eulerian fields

bias_terms_eul_pred=[]
for ii in range(0,len(bias_fields)):
    bias_terms_pred = bacco.statistics.compute_mesh(ngrid=npart, box=boxsize, pos=pred_pos, 
                              mass = (bias_fields[ii]).flatten(), deposit_method='cic', 
                              interlacing=False)
    bias_terms_eul_pred.append(bias_terms_pred)
bias_terms_eul_pred = np.array(bias_terms_eul_pred)

## Save eulerian fields

np.save('Eulerian_fields.npy', bias_terms_eul_pred, allow_pickle=True)

