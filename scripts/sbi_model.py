import numpy as np
import os
import pathlib
import pickle
import wandb

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
                     param_names=None, dict_bounds=None, run_mode='single',
                     tag_sbi='', n_threads=8,
                     sweep_name=None,
                     ):
        
        # if n_threads is not None:
        #     tf.config.threading.set_inter_op_parallelism_threads(n_threads)
        #     tf.config.threading.set_intra_op_parallelism_threads(n_threads)
        
        # TODO the error is not being used!!! ack! write custom loss func?
        # actually unclear how the error should be taken into account... 

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
        
        #if training, need to pass param names
        #if run_mode != 'load':
        # ideally for testing, the saved SBI model would know the 
        # parameter names, but it seems that they can't save them
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
            
        # If we don't have training data, we'll probs want the scaler
        # (but maybe there's a better place for this...?)
        if self.y_train_unscaled is None:
            self.load_scaler_y()
            
        if self.theta_train is not None:
            self.n_params = self.theta_train.shape[1]
        elif self.theta_test is not None:
            self.n_params = self.theta_test.shape[1]
        
        self.run_mode = run_mode
        if self.run_mode != 'load':
            assert dict_bounds is not None, 'need dict_bounds if training model'
        self.dict_bounds = dict_bounds
        self.sweep_name = sweep_name

        
    def run(self, max_epochs=1000):
        
        print("run mode:", self.run_mode)
        print("sweep name:", self.sweep_name)
        print("dir_sbi:", self.dir_sbi)
        
        project_name = 'muchisimocks-sbi'
        
        if self.run_mode == 'sweep':
            count = 10  # number of runs in sweep for random search
            sweep_config = {
                'name': self.sweep_name,
                'method': 'random',
                'metric': {
                    'name': 'validation_loss',
                    'goal': 'minimize'
                },
                'parameters': {
                    'learning_rate': {'values': [1e-2, 1e-3, 1e-4, 1e-5]},
                    'hidden_features': {'values': [32, 64, 128]},
                    'training_batch_size': {'values': [32, 64, 128]},
                    'model_type': {'values': ['maf']},
                    #'model_type': {'values': ['maf', 'nsf']},
                    'max_epochs': {'value': max_epochs},
                }
            }
            sweep_id = wandb.sweep(sweep_config, project=project_name)
            
            def _fit_model_wandb():
                wandb.init()
                self.fit_model(wandb.config, save_model=False)
                
            wandb.agent(sweep_id, function=_fit_model_wandb, count=count)
            wandb.finish()
            
        elif self.run_mode == 'best':
            wandb.login()
            api = wandb.Api()
            # get sweep name
            sweeps = api.project(project_name).sweeps()
            
            self.sweep = next((s for s in sweeps if s.name == self.sweep_name), None)
            assert self.sweep is not None, f"Sweep {self.sweep_name} not found"
            
            # Get best run parameters
            best_run = self.sweep.best_run()
            print(f"Using hyperparameters from best run from sweep {self.sweep_name}: {best_run.config}")
            wandb.init(project=project_name, config=best_run.config)
            self.fit_model(wandb.config, save_model=True)
            wandb.finish()
            
        elif self.run_mode == 'single':
            # Use default config for single run
            config = {
                'learning_rate': 1e-3,
                'hidden_features': 128,
                'training_batch_size': 64,
                'model_type': 'maf',
                'max_epochs': max_epochs,
            }
            wandb.init(project=project_name, config=config)
            self.fit_model(wandb.config, save_model=True)
            wandb.finish()

        elif self.run_mode == 'load':
            self.load_posterior()
            self.load_param_names()
            # may need to get n_params somehow
            
        else:
            raise ValueError(f"run_mode {self.run_mode} not recognized")
            
    def fit_model(self, config, save_model=True):
        """
        Fit the SBI model using the specified configuration
        """
        
        print(f"Fitting model for dir_sbi={self.dir_sbi}, run_mode={self.run_mode}, sweep_name={self.sweep_name}")
        print("wandb.config:", config)
        
        # get prior
        l_bounds = np.array([self.dict_bounds[pn][0] for pn in self.param_names])
        u_bounds = np.array([self.dict_bounds[pn][1] for pn in self.param_names])
        prior = BoxUniform(low=torch.from_numpy(l_bounds),
                           high=torch.from_numpy(u_bounds))

        print(self.dict_bounds)
        print("Setting up inference")
        
        # Pull hyperparameters from config
        learning_rate = config.learning_rate
        training_batch_size = config.training_batch_size
        hidden_features = config.hidden_features
        model_type = config.model_type
        max_epochs = config.max_epochs
            
        density_estimator_build_fun = posterior_nn(
            model=model_type,
            hidden_features=hidden_features,
            # num_transforms and num_blocks could also be hyperparameters
        )

        # can't pass in own validation set without writing custom training loop,
        # so doing this hack (don't even know if the fraction is taken from the end
        # or randomly, so may be mixing train and val here)
        theta_train_and_val = np.concatenate((self.theta_train, self.theta_val), axis=0)
        y_train_and_val = np.concatenate((self.y_train, self.y_val), axis=0)
        validation_fraction = len(self.theta_val) / len(theta_train_and_val)
        print(self.theta_val.shape, self.theta_train.shape)
        print(f"Validation fraction: {validation_fraction}")
        print(self.theta_val)
        
        inference = NPE(prior=prior, density_estimator=density_estimator_build_fun)
        inference = inference.append_simulations(
            torch.tensor(theta_train_and_val, dtype=torch.float32),
            torch.tensor(y_train_and_val, dtype=torch.float32),
            )
        
        print("Training")
        # Define a custom callback for wandb logging
        # class WandbLoggingCallback(sbi.inference.callbacks.IterationCallback):
        #     def __call__(self, epoch, train_log_prob, validation_log_prob):
        #         # Log to wandb
        #         wandb.log({
        #             "epoch": epoch,
        #             "training_loss": -train_log_prob,  # negative log likelihood
        #             "validation_loss": -validation_log_prob  # negative log likelihood
        #         })
        #         return True  # Continue training
                
        #callbacks = [WandbLoggingCallback()]
        
        density_estimator = inference.train(
            max_num_epochs=max_epochs,
            training_batch_size=training_batch_size,
            validation_fraction=validation_fraction,
            learning_rate=learning_rate,
            show_train_summary=True,
            #callbacks=callbacks
            )
        print("Trained!")

        # Get the final training and validation losses
        train_log = inference._summary
        if train_log and len(train_log['training_loss']) > 0:
            final_training_loss = train_log['training_loss'][-1] if train_log['training_loss'] is not None else None
            final_validation_loss = train_log['validation_loss'][-1] if train_log['validation_loss'] is not None else None
            #final_training_loss = train_log[0].get("training_log_probs", [])[-1] if train_log[0].get("training_log_probs", []) else None
            #final_validation_loss = train_log[0].get("validation_log_probs", [])[-1] if train_log[0].get("validation_log_probs", []) else None
            
            if final_training_loss is not None:
                wandb.log({"final_training_loss": final_training_loss})
            if final_validation_loss is not None:
                wandb.log({"final_validation_loss": final_validation_loss})
        
        print("Building posterior")
        self.posterior = inference.build_posterior(density_estimator)
        print(self.posterior)
        
        # save model if requested (e.g., for best model from sweep or single run)
        if save_model:
            with open(f"{self.dir_sbi}/posterior.p", "wb") as f:
                pickle.dump(self.posterior, f)
            with open(f"{self.dir_sbi}/inference.p", "wb") as f:
                pickle.dump(inference, f)
            with open(f"{self.dir_sbi}/param_names.txt", "w") as f:
                np.savetxt(f, self.param_names, fmt="%s")
            
            # Also save the hyperparameter configuration
            config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
            with open(f"{self.dir_sbi}/config.pkl", "wb") as f:
                pickle.dump(config_dict, f)
            
            print(f"Saved model to {self.dir_sbi}")

            
    def load_posterior(self):
        fn_posterior = f'{self.dir_sbi}/posterior.p'
        assert os.path.exists(fn_posterior), f"posterior.p not found in {self.dir_sbi}"
        print(f"Loading posterior from {fn_posterior}")
        with open(fn_posterior, "rb") as f:
            self.posterior = pickle.load(f)

    def load_param_names(self):
        fn_param_names = f'{self.dir_sbi}/param_names.txt'
        assert os.path.exists(fn_param_names), f"param_names.txt not found in {self.dir_sbi}"
        print(f"Loading param_names from {fn_param_names}")
        with open(fn_param_names, "r") as f:
            self.param_names = np.loadtxt(f, dtype=str)

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
        
    
    def evaluate(self, y_obs_unscaled, n_samples=10000):
        # convergence tests show 10,000 is probably good enough, tho for some
        # parameters there is fluctuation bw 10k, 30k, 100k
        # (see notebooks/2025-01-24_inference_muchisimocksPk.ipynb)
        y_obs = self.scaler_y.scale(y_obs_unscaled)
        # model is built with float32 so need the data to be here too
        y_obs = np.float32(y_obs)
        if y_obs.ndim == 1:
            samples = self.posterior.sample((n_samples,), x=y_obs)
        elif y_obs.ndim == 2:
            samples = self.posterior.sample_batched((n_samples,), x=y_obs)
        else:
            raise ValueError(f"y_obs has wrong number of dims! should be 1 or 2 (y_obs.ndim)={y_obs.ndim}")
        return samples
    
    
    def evaluate_test_set(self, y_test_unscaled=None, tag_test=''):
        if y_test_unscaled is None:
            y_test_unscaled = self.y_test_unscaled
        samples_test_pred = self.evaluate(y_test_unscaled)
        fn_samples_test_pred = f'{self.dir_sbi}/samples_test{tag_test}_pred.npy'
        np.save(fn_samples_test_pred, samples_test_pred)
        print(f"Saved samples to {fn_samples_test_pred}")