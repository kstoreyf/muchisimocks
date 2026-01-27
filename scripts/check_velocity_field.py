import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import pandas as pd
import pickle

import bacco
import baccoemu

import sys
sys.path.append('/dipc/kstoreyf/muchisimocks/scripts')
import utils
import plotter

n_grid = 512
box_size = 1000.

# ## Load in z=0 quijote LR snapshot 663
# idx_LH_str = '0663'
# sim_name_quijote = f'quijote_LH{idx_LH_str}'

# #dir_data = '/cosmos_storage/home/mpelle/Yin_data/Quijote'
# dir_mocks = '/scratch/kstoreyf/Yin_data/Quijote' #hyp
# fn_params = f'{dir_mocks}/LH{idx_LH_str}/param_{idx_LH_str}.txt'
# param_vals = np.loadtxt(fn_params)
# param_names = ['omega_m', 'omega_baryon', 'h', 'n_s', 'sigma_8']
# param_dict = dict(zip(param_names, param_vals))
# param_dict['tau'] = 0.0952 # ?? TODO check proper tau to be using!! 
# cosmo_quijote = utils.get_cosmo(param_dict)
# import readgadget

idx_LH = '663'
# #snapshot = '/dipc/kstoreyf/Quijote_simulations/Snapshots/latin_hypercube/663/snapdir_004/snap_004' # 004 = z0 #atlas248
# snapshot = f'/scratch/kstoreyf/Quijote_simulations/Snapshots/latin_hypercube/{idx_LH}/snapdir_004/snap_004' # 004 = z0 #hyperion
# ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)


# # read positions, velocities and IDs of the particles
# pos_snap = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
# vel_snap = readgadget.read_block(snapshot, "VEL ", ptype)
# print(pos_snap.shape, vel_snap.shape)
# ids = readgadget.read_block(snapshot, "ID  ", ptype)-1   #IDs starting from 0
# # ics
# snapshot_ics = '/scratch/kstoreyf/Quijote_simulations/Snapshots/latin_hypercube/663/ICs/ics'
# ids_ics = readgadget.read_block(snapshot_ics, "ID  ", ptype)-1   #IDs starting from 0

# fn_lag_index = f'/scratch/kstoreyf/Quijote_simulations/quijote_LH{idx_LH}_neighfile.pickle'
# with open(fn_lag_index, 'rb') as f:
#     lag_index = pickle.load(f)
# print(lag_index.shape)
# pos_snap_ord = pos_snap[lag_index]
# vel_snap_ord = vel_snap[lag_index]

# ## Check velocity smoothing
# ### compute bispec
# # trying this

# mesh = bacco.statistics.compute_mesh(ngrid=n_grid, box=box_size, pos=pos_snap, 
#                                      vel=None, mass=None,
#                  interlacing=False, deposit_method='cic',
#                  zspace=False, cosmology=None)
# print(mesh.shape)

# fn_mesh = f'/scratch/kstoreyf/Quijote_simulations/quijote_LH{idx_LH}_mesh.npy'
# np.save(fn_mesh, mesh[0])


fn_mesh = f'/scratch/kstoreyf/Quijote_simulations/quijote_LH{idx_LH}_mesh.npy'
mesh = np.load(fn_mesh)

import compute_statistics as cs
# compute bispectrum
n_threads = 8
base = cs.setup_bispsec(box_size, n_grid, n_threads)

bspec, bk_corr = cs.compute_bispectrum(base, mesh)
k123 = bspec.get_ks()
weight = k123.prod(axis=0)
bispec_results_dict = {
    'k123': k123,
    'bispectrum': bk_corr,
    'weight': weight,
}

fn_bispec = f'/scratch/kstoreyf/Quijote_simulations/bispsec_quijote_LH{idx_LH}.npy'
np.save(fn_bispec, bispec_results_dict)
