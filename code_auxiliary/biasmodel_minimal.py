import bacco

import numpy as np


def main():
    print("Setting up bias model", flush=True)
    n_grid = 32
    dens_lin = np.ones((n_grid, n_grid, n_grid))
    damping_scale = 0.2
    interlacing = False
    box_size = 100.
    bmodel = bacco.BiasModel(sim=None, linear_delta=dens_lin, BoxSize=box_size,
                            ngrid=n_grid, ngrid1=None,
                            sdm=False, mode="dm",
                            npart_for_fake_sim=n_grid, damping_scale=damping_scale,
                            bias_model='expansion', deposit_method="cic",
                            use_displacement_of_nn=False, interlacing=interlacing,
                            )

    ## Compute lagrangian fields
    print("Computing lagrangian fields", flush=True)
    bias_fields = bmodel.bias_terms_lag()
    print("Done!")


if __name__=='__main__':
    main()
