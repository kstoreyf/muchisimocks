import os

# # Set environment variables before importing torch
# os.environ['OMP_NUM_THREADS'] = '24'
# os.environ['MKL_NUM_THREADS'] = '24'
# os.environ['NUMEXPR_NUM_THREADS'] = '24'

import numpy as np
import pathlib
import pickle
import time
import wandb

import sbi
from sbi.inference import NPE
from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn
import torch

import scaler_custom as scl
import generate_params as genp
import utils


class SBIModel():

    def __init__(self, theta_train=None, y_train_unscaled=None, y_err_train_unscaled=None,
                     theta_val=None, y_val_unscaled=None, y_err_val_unscaled=None,
                     theta_test=None, y_test_unscaled=None, y_err_test_unscaled=None,
                     statistics=None, param_names=None, dict_bounds=None, 
                     run_mode='single', tag_sbi='', n_threads=1, 
                     sweep_name=None,
                     ):
        
        # training does not seem to be parallelizeable! getting no speedup
        # if n_threads is not None:
        #     tf.config.threading.set_inter_op_parallelism_threads(n_threads)
        #     tf.config.threading.set_intra_op_parallelism_threads(n_threads)
        # NOTE the error is not used in SBI

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
        assert len(statistics) > 0, 'Pass statistics! (Needed for scaler)'
        self.statistics = statistics
        
        if self.y_train_unscaled is not None:
            self.setup_scalers_y()
            self.n_dim = self.y_train.shape[1]
            #self.n_dim = self.y_train.shape[1:]
            print('ndim:', self.n_dim)
        elif self.y_test_unscaled is not None:
            # better way to do this now that might have multiple statistics?
            self.n_dim = np.sum([y_test_i.shape[1] for y_test_i in self.y_test_unscaled])
            #self.n_dim = self.y_test_unscaled.shape[1]
            #self.n_dim = self.y_test.shape[1:]
            
        # If we don't have training data, we'll probs want the scaler
        # (but maybe there's a better place for this...?)
        if self.y_train_unscaled is None:
            self.load_scalers_y()
            
        if self.theta_train is not None:
            self.n_params = self.theta_train.shape[1]
        elif self.theta_test is not None:
            self.n_params = self.theta_test.shape[1]
        
        self.run_mode = run_mode
        if self.run_mode != 'load':
            assert dict_bounds is not None, 'need dict_bounds if training model'
        self.dict_bounds = dict_bounds
        self.sweep_name = sweep_name
        
        self.n_threads = n_threads

        
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
        
         # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Optimize PyTorch settings
        if device == "cpu":
            print(f"Using CPU with {self.n_threads} threads")
            torch.set_num_threads(self.n_threads)  # Use multiple CPU threads
        elif device == "cuda":
            print(f"Using GPU with {torch.cuda.device_count()} devices")
        
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
        print('theta_train shape:', self.theta_train.shape, 'theta_val shape:', self.theta_val.shape)
        print(f"Validation fraction: {validation_fraction}")
        
        inference = NPE(prior=prior, density_estimator=density_estimator_build_fun)
        inference = inference.append_simulations(
            torch.tensor(theta_train_and_val, dtype=torch.float32),
            torch.tensor(y_train_and_val, dtype=torch.float32),
            )
        
        print(f"Training with {self.n_threads} threads")
        start = time.time()
        
        # defaults: 
        # stop_after_epochs=20
        # num_atoms = 10
        density_estimator = inference.train(
            max_num_epochs=max_epochs,
            training_batch_size=training_batch_size,
            validation_fraction=validation_fraction,
            learning_rate=learning_rate,
            show_train_summary=True,
            #callbacks=callbacks
            )
        print("Trained!")
        end = time.time()
        print(f"Training time: {end - start:.2f}s = {(end - start) / 60:.2f} min (max_epochs={max_epochs}, n_threads={self.n_threads})")

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


    def setup_scalers_y(self):
        
        self.scalers_y = []
        # these have length the number of data samples
        self.y_train = np.empty((len(self.y_train_unscaled[0]), 0))
        self.y_val = np.empty((len(self.y_val_unscaled[0]), 0))
        print(f"y_train shape: {self.y_train.shape}")
                
        for i, statistic in enumerate(self.statistics):
            
            func_scaler_y = utils.statistics_scaler_funcs[statistic]
            
            scaler_y = scl.Scaler(func_scaler_y)
            print("statistic:", statistic)
            print(f"min and max before scaling: {np.min(self.y_train_unscaled[i]):3f}, {np.max(self.y_train_unscaled[i]):3f}")
            
            scaler_y.fit(self.y_train_unscaled[i])
            self.scalers_y.append(scaler_y)
            
            y_train_i = scaler_y.scale(self.y_train_unscaled[i])
            y_val_i = scaler_y.scale(self.y_val_unscaled[i])

            self.y_train = np.concatenate((self.y_train, y_train_i), axis=1)
            self.y_val = np.concatenate((self.y_val, y_val_i), axis=1)
            if self.y_test_unscaled is not None:
                y_test_i = scaler_y.scale(self.y_test_unscaled[i])
                self.y_test = np.concatenate((self.y_test, y_test_i), axis=1)
            
            print(f"min and max after scaling: {np.min(scaler_y.scale(self.y_train_unscaled[i])):3f}, {np.max(scaler_y.scale(self.y_train_unscaled[i])):3f}")
            
            # save scaler - need pickle for custom object!!
            fn_scaler_y = f'{self.dir_sbi}/scaler_y_{statistic}.p'
            with open(fn_scaler_y, "wb") as f:
                pickle.dump(scaler_y, f)
                
        if self.y_test_unscaled is not None:
            print(f"y_test shape: {self.y_test.shape}")
                

    def load_scalers_y(self):

        self.scalers_y = []
        for i, statistic in enumerate(self.statistics):
            fn_scaler_y = f'{self.dir_sbi}/scaler_y_{statistic}.p'
            with open(fn_scaler_y, "rb") as f:
                self.scalers_y.append(pickle.load(f))
        print(f"Loaded scalers from {self.dir_sbi}")
        
    
    def evaluate(self, y_obs_unscaled, n_samples=10000):
        # convergence tests show 10,000 is probably good enough, tho for some
        # parameters there is fluctuation bw 10k, 30k, 100k
        # (see notebooks/2025-01-24_inference_muchisimocksPk.ipynb)
        if y_obs_unscaled[0].ndim == 1:
            n_data = 1
        else:
            n_data = y_obs_unscaled[0].shape[0]
        y_obs = np.empty((n_data, 0))
        for i, y_obs_unscaled_i in enumerate(y_obs_unscaled):
            if y_obs_unscaled[0].ndim == 1:
                y_obs_unscaled_i = np.expand_dims(y_obs_unscaled_i, axis=0)
            y_test_i = self.scalers_y[i].scale(y_obs_unscaled_i)
            y_obs = np.concatenate((y_obs, y_test_i), axis=1
                                   )
        print(f"Testing on y_obs with shape: {y_obs.shape}")
        start = time.time()
        # model is built with float32 so need the data to be here too
        y_obs = np.float32(np.array(y_obs))
        # using samples_batched bc always putting into 2d first (if were 2d, "samples")
        
        samples = self.posterior.sample_batched((n_samples,), x=y_obs)
        print(f"Time to sample (y_obs.shape={y_obs.shape}, n_samples={n_samples}): {time.time() - start:.2f}s = {(time.time() - start) / 60:.2f} min")
        return samples
    
    
    
    def evaluate_test_set(self, y_test_unscaled=None, tag_test='', 
                          n_samples=10000, checkpoint_every=100, 
                          #n_samples=200, checkpoint_every=10, 
                          resume=True):
        
        ### NOTE: this went orders of mag faster when i added checkpointing every 100 and doing 
        # samples_batched of that size! before sometimes would only finish a 20-50% in a day;
        # now finishing in 4-8 hours, for 1000 test set with 10000 samples
        
        # y_test_unscaled is an array of length n_statistics, each with shape (n_test, n_dim);
        # concatenate inside evaluate bc we need to scale based on each stat
        print(f"Evaluating test set with tag {tag_test}")
        if y_test_unscaled is None:
            y_test_unscaled = self.y_test_unscaled
        
        # Set up file paths
        fn_samples_test_pred = f'{self.dir_sbi}/samples_test{tag_test}_pred.npy'
        fn_samples_test_pred_inprogress = f'{self.dir_sbi}/samples_test{tag_test}_pred_inprogress.npy'
        checkpoint_file = f"{self.dir_sbi}/checkpoint_samples_test{tag_test}.txt"
        
        # Check for existing samples and checkpoint
        samples_total = len(y_test_unscaled[0])
        samples_completed = 0
        existing_samples = None
        
        print(f"Checkpoint file: {checkpoint_file}")
        
        if resume:
            # Check if final file already exists (complete run)
            if os.path.exists(fn_samples_test_pred):
                existing_samples = np.load(fn_samples_test_pred)
                if existing_samples.shape[0] >= samples_total:
                    print(f"Found complete samples file: {fn_samples_test_pred} with {existing_samples.shape[0]} samples")
                    return
            
            # Check existing in-progress samples file
            if os.path.exists(fn_samples_test_pred_inprogress):
                existing_samples = np.load(fn_samples_test_pred_inprogress)
                samples_completed = existing_samples.shape[0]
                print(f"Found existing in-progress samples file with {samples_completed} samples")
                
            # Check checkpoint file for consistency
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    checkpoint_count = int(f.read().strip())
                print(f"Checkpoint file indicates {checkpoint_count} completed samples")
                
                # Use the checkpoint count if consistent, otherwise trust the samples file
                if existing_samples is not None and checkpoint_count == existing_samples.shape[0]:
                    samples_completed = checkpoint_count
                elif existing_samples is not None:
                    print(f"Checkpoint mismatch - using samples file count: {existing_samples.shape[0]}")
                    samples_completed = existing_samples.shape[0]
                else:
                    samples_completed = checkpoint_count
            
            if samples_completed >= samples_total:
                print(f"All {samples_total} samples already completed!")
                return
                
            if samples_completed > 0:
                print(f"Resuming from {samples_completed} completed samples")
        
        start_time = time.time()
        
        # Sample in batches
        remaining_samples = samples_total - samples_completed
        
        try:
            while remaining_samples > 0:
                batch_size = min(checkpoint_every, remaining_samples)
                print(f"Sampling batch of {batch_size} samples ({samples_completed}/{samples_total} completed)")
                
                # Extract the chunk of observations we need to process
                start_idx = samples_completed
                end_idx = samples_completed + batch_size
                
                # Get the batch of y_test_unscaled data for this chunk
                if y_test_unscaled[0].ndim == 1:
                    # Single observation case - just use the same observation for all samples
                    y_test_unscaled_batch = y_test_unscaled
                else:
                    # Multiple observations case - extract the chunk from each statistic's array
                    y_test_unscaled_batch = [y_stat[start_idx:end_idx] for y_stat in y_test_unscaled]
                
                batch_start = time.time()
                # Use the existing evaluate method for this batch
                print(f"Evaluating batch {start_idx} to {end_idx}")
                batch_samples = self.evaluate(y_test_unscaled_batch, n_samples=n_samples)
                batch_end = time.time()
                
                print(f"Batch samples shape: {batch_samples.shape}")
                
                # Combine with existing samples if any (concatenate along axis=1 for test observations)
                if existing_samples is not None:
                    current_samples = np.concatenate([existing_samples, batch_samples], axis=1)
                else:
                    current_samples = batch_samples
                print(f"Current samples shape: {current_samples.shape}")
                
                # Save updated samples to in-progress file
                np.save(fn_samples_test_pred_inprogress, current_samples)
                
                # Update counts
                samples_completed += batch_size
                remaining_samples -= batch_size
                existing_samples = current_samples
                
                # Save simple text checkpoint
                with open(checkpoint_file, 'w') as f:
                    f.write(str(samples_completed))
                
                print(f"Batch completed in {batch_end - batch_start:.2f}s ({(batch_end - batch_start) / 60:.2f} min) ({(batch_end - batch_start) / 3600:.2f} hrs")
                print(f"Saved {samples_completed}/{samples_total} samples")
                
        except Exception as e:
            print(f"Error during sampling: {e}")
            print(f"Partial results saved: {samples_completed}/{samples_total} samples")
            print(f"Resume by running again - will continue from {samples_completed} samples")
            print(f"In-progress file: {fn_samples_test_pred_inprogress}")
            raise
        
        end_time = time.time()
        print(f"Total sampling time (n_samples={n_samples} per obs): {end_time - start_time:.2f}s = {(end_time - start_time) / 60:.2f} min")
        
        # Move in-progress file to final file when complete
        if os.path.exists(fn_samples_test_pred_inprogress):
            os.rename(fn_samples_test_pred_inprogress, fn_samples_test_pred)
            print(f"Sampling complete! Moved to final file: {fn_samples_test_pred}")
        
        # Clean up checkpoint file on successful completion
        # if os.path.exists(checkpoint_file):
        #     os.remove(checkpoint_file)
        #     print("Checkpoint file removed after successful completion")