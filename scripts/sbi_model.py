import numpy as np
import os
import pathlib
import pickle 

import sbi
from sbi.inference import NPE
from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn
import torch

import scaler_custom as scl
import generate_params_lh as gplh


class SBIModel(): 

    def __init__(self, theta_train=None, y_train_unscaled=None, y_err_train_unscaled=None,
                       theta_val=None, y_val_unscaled=None, y_err_val_unscaled=None,
                       theta_test=None, y_test_unscaled=None, y_err_test_unscaled=None,
                       param_names=None, run_mode='single',
                       tag_sbi='', n_threads=8, 
                       ):
        
        # if n_threads is not None:
        #     tf.config.threading.set_inter_op_parallelism_threads(n_threads)
        #     tf.config.threading.set_intra_op_parallelism_threads(n_threads)
        # # TODO error not being used!!! ack! write custom loss func

        self.dir_sbi = f'../results/results_sbi/sbi{tag_sbi}'
        p = pathlib.Path(self.dir_sbi)
        p.mkdir(parents=True, exist_ok=True)
        
        self.theta_train = theta_train
        self.y_train_unscaled = y_train_unscaled
        self.y_err_train_unscaled = y_err_train_unscaled

        self.theta_val = theta_val
        self.y_val_unscaled = y_val_unscaled
        self.y_err_val_unscaled = y_err_val_unscaled

        self.theta_test = theta_test
        self.y_test_unscaled = y_test_unscaled
        self.y_err_test_unscaled = y_err_test_unscaled
        
        assert param_names is not None, 'need parameter names'
        self.param_names = param_names
        
        if self.y_train_unscaled is not None:
            self.setup_scaler_y()
            self.n_dim = self.y_train.shape[1]
            #self.n_dim = self.y_train.shape[1:]
            print(self.n_dim)
        elif self.y_test_unscaled is not None:
            self.n_dim = self.y_test_unscaled.shape[1]
            #self.n_dim = self.y_test.shape[1:]
            
        # If we don't have training data, we'll probs want the scale
        # (but maybe there's a better place for this...?)
        if self.y_train_unscaled is None:
            self.load_scaler_y()
            
        if self.theta_train is not None:
            self.n_params = self.theta_train.shape[1]
        elif self.theta_test is not None:
            self.n_params = self.theta_test.shape[1]
        
        self.run_mode = run_mode

        
    def run(self, max_epochs=20):

        if self.run_mode == 'single':
            # get prior
            _, bounds_dict, _ = gplh.define_LH_cosmo()
            l_bounds = np.array([bounds_dict[pn][0] for pn in self.param_names])
            u_bounds = np.array([bounds_dict[pn][1] for pn in self.param_names])
            prior = BoxUniform(low=torch.from_numpy(l_bounds), 
                                        high=torch.from_numpy(u_bounds))

            print("Setting up inference")
            # SimBIG switched to NSF (neural spine flow) from originally an MAF (masked autoregressive flow)
            # could customize this further (that's where would tune hyperparams) 
            # https://sbi-dev.github.io/sbi/latest/tutorials/03_density_estimators/#changing-hyperparameters-of-density-estimators
            # SimBIG uses an ensemble of 5 NSFs
            
                
            density_estimator_build_fun = posterior_nn(
                model='maf',
                # hidden_features=num_hidden_features,
                # num_transforms=num_transforms,
                # num_blocks=num_blocks
            )
    
            theta_train_and_val = np.concatenate((self.theta_train, self.theta_val), axis=0)
            y_train_and_val = np.concatenate((self.y_train, self.y_val), axis=0)
            validation_fraction = len(self.theta_val) / len(theta_train_and_val)
            print(f"Validation fraction: {validation_fraction}")
    
            inference = NPE(prior=prior, density_estimator=density_estimator_build_fun)
            inference = inference.append_simulations(
                torch.tensor(theta_train_and_val, dtype=torch.float32),
                torch.tensor(y_train_and_val, dtype=torch.float32),
                )
            
            
            
            print("Training")
            density_estimator = inference.train(
                max_num_epochs=max_epochs,
                # training_batch_size=training_batch_size,
                validation_fraction=validation_fraction,
                # learning_rate=learning_rate,
                show_train_summary=True
                )
            
            print("Building posterior")
            self.posterior = inference.build_posterior(density_estimator)
            print(self.posterior)
            with open(f"{self.dir_sbi}/posterior.p", "wb") as f:
                pickle.dump(self.posterior, f)

        elif self.run_mode == 'load':
            self.load_posterior()
            # may need to get n_params somehow
            
        else: 
            raise ValueError("run_mode not recognized")
            
            
    def load_posterior(self):
        fn_posterior = f'{self.dir_sbi}/posterior.p'
        assert os.path.exists(fn_posterior), f"model_mean.keras not found in {self.dirsbi}"
        print(f"Loading mean model from {fn_posterior}")
        with open(fn_posterior, "rb") as f:
            self.posterior = pickle.load(f)


    def setup_scaler_y(self):
        self.scaler_y = scl.Scaler('log_minmax')
        self.scaler_y.fit(self.y_train_unscaled)
        
        self.y_train = self.scaler_y.scale(self.y_train_unscaled)
        self.y_val = self.scaler_y.scale(self.y_val_unscaled)
        if self.y_test_unscaled is not None:
            self.y_test = self.scaler_y.scale(self.y_test_unscaled)
        
        fn_scaler_y = f'{self.dir_sbi}/scaler_y.p'
        
        # need pickle for custom object!!
        with open(fn_scaler_y, "wb") as f:
            pickle.dump(self.scaler_y, f)


    def load_scaler_y(self):
        fn_scaler_y = f'{self.dir_sbi}/scaler_y.p'
        #self.scaler_cov = np.load(fn_scaler_cov, allow_pickle=True)
        with open(fn_scaler_y, "rb") as f:
            self.scaler_y = pickle.load(f)
        
        
    def evaluate(self, y_obs_unscaled, n_samples=1000, mean_only=False):
        samples = self.posterior.sample((n_samples,), x=y_obs_unscaled)
        return samples
    
    
    def evaluate_test_set(self, y_test_unscaled=None, theta_test=None, tag_test=''):
        if y_test_unscaled is None:
            y_test_unscaled = self.y_test_unscaled
        samples_test_pred = self.evaluate(y_test_unscaled)
        fn_samples_test_pred = f'{self.dir_sbi}/samples_test{tag_test}_pred.npy'
        np.save(fn_samples_test_pred, samples_test_pred)
        print(f"Saved samples to {fn_samples_test_pred}")