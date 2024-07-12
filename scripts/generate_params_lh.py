import os
import numpy as np

from scipy.stats import qmc


nsamples = 10000 #500

OmegaBaryon = 0.049
ns = 0.9624

sampler = qmc.LatinHypercube(d=3)
sample = sampler.random(n=nsamples)

l_bounds = [0.23, 0.65, 0.6]
u_bounds = [0.4, 0.9, 0.8]
sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

for ii in range(nsamples):
    cosparams = np.hstack((sample_scaled[ii],[OmegaBaryon],[ns],[ii]))
    os.system('mkdir ../../../cosmolib/LH'+str(ii))
    np.savetxt('../../../cosmolib/LH'+str(ii)+'/cosmo_'+str(ii)+'.txt',cosparams.T, fmt='%s')