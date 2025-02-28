import numpy as np
import pickle
import readgadget

from pathlib import Path


# Adapted from Raul Angulo's notebook QuijoteBias.ipynb

ngrid = 512
idx_LH = 663
BoxSize = 1000
ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)

snapnums = ['000', '001', '002', '003', '004']

for snapnum in snapnums:
    print(snapnum)

    snapshot = f'/scratch/kstoreyf/Quijote_simulations/Snapshots/latin_hypercube/{idx_LH}/snapdir_{snapnum}/snap_{snapnum}' # 004 = z0 #hyperion
    snapshot_ics = f'/scratch/kstoreyf/Quijote_simulations/Snapshots/latin_hypercube/{idx_LH}/ICs/ics'


    # read the positions and IDs of the ICs
    pos_ICs = readgadget.read_block(snapshot_ics, "POS ", ptype)/1e3 #Mpc/h
    IDs_ICs = readgadget.read_block(snapshot_ics, "ID  ", ptype)-1   #IDs begin from 0

    # sort the ICs particles by IDs
    indexes = np.argsort(IDs_ICs)
    pos_ICs = pos_ICs[indexes]; 
    #del IDs_ICs

    # read the positions and IDs of the z=0 snapshot
    #pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #Mpc/h
    IDs = readgadget.read_block(snapshot, "ID  ", ptype)-1   #Make IDs begin from 0

    # find the grid coordinates of the particles
    grid_index = (np.round((pos_ICs/BoxSize)*ngrid, decimals=0)).astype(np.int32)
    grid_index[np.where(grid_index==ngrid)]=0
    pos_lag    = grid_index*BoxSize/ngrid #get the lagrangian coordinates
    grid_index = grid_index[:,0]*ngrid**2 + grid_index[:,1]*ngrid + grid_index[:,2]
    indexes2   = np.argsort(grid_index)
    # sort the particles by IDs
    indexes = np.argsort(IDs)
    lag_index = indexes[indexes2]


    #Saving indexes 
    fn_lag_index = f'/scratch/kstoreyf/Quijote_simulations/quijote_LH{idx_LH}_snap{snapnum}_neighfile.pickle'
    with open(fn_lag_index, 'wb') as f:
        pickle.dump(lag_index, f, protocol=-1)

    fn_lag_index_ics = f'/scratch/kstoreyf/Quijote_simulations/quijote_ics_LH{idx_LH}_neighfile.pickle'
    if Path(fn_lag_index_ics).exists():
        print(f"IC neighfile {fn_lag_index_ics} already exists! skipping")
    else:
        indexes = np.argsort(IDs_ICs)
        lag_index_ics = indexes[indexes2]
        # safety check
        grid_index = grid_index[indexes2]
        diff = grid_index[1:] - grid_index[:-1]
        if np.any(diff!=1):  raise Exception('positions not properly sorted')

        with open(fn_lag_index_ics, 'wb') as f:
            pickle.dump(lag_index_ics, f, protocol=-1)
