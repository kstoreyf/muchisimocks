import numpy as np
import os
import pathlib
import pickle
import tensorflow as tf
import wandb

from tensorflow.keras.callbacks import ModelCheckpoint

import neural_network as nn
import scaler_custom as scl




class MomentNetwork(): 

    def __init__(self, theta_train=None, y_train_unscaled=None, y_err_train_unscaled=None,
                       theta_val=None, y_val_unscaled=None, y_err_val_unscaled=None,
                       theta_test=None, y_test_unscaled=None, y_err_test_unscaled=None,
                       tag_mn='', n_threads=8, cov_mode='direct',
                       run_mode_mean='best', run_mode_cov=None,
                       sweep_name_mean=None, sweep_name_cov=None,
                       ):
        
        if n_threads is not None:
            tf.config.threading.set_inter_op_parallelism_threads(n_threads)
            tf.config.threading.set_intra_op_parallelism_threads(n_threads)
        # TODO error not being used!!! ack! write custom loss func

        self.dir_mn = f'../results/results_moment_network/mn{tag_mn}'
        p = pathlib.Path(self.dir_mn)
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
        
        self.cov_mode = cov_mode
        self.include_covariances = True
        self.run_mode_mean = run_mode_mean
        self.run_mode_cov = run_mode_cov
        self.sweep_name_mean = sweep_name_mean
        self.sweep_name_cov = sweep_name_cov
        


    def run(self, max_epochs_mean=1500, max_epochs_cov=1000):
        
        print("run mode mean:", self.run_mode_mean)
        print("run mode cov:", self.run_mode_cov)
        print("sweep name mean:", self.sweep_name_mean)
        print("sweep name cov:", self.sweep_name_cov)
        print(("dir_mn:", self.dir_mn))
        
        ### Mean model ###
        
        project_name_mean = 'muchisimocks-mn-mean'
        
        if self.run_mode_mean == 'sweep':
            count = 10 #if random
            sweep_config = {
                'name': self.sweep_name_mean,
                #'method': 'grid',
                'method': 'random',
                'metric': {
                    'name': 'val_loss',
                    'goal': 'minimize'
                },
                'parameters': {
                    'learning_rate': {'values': [1e-2, 1e-3, 1e-4, 1e-5]},
                    'hidden_size': {'values': [32, 64, 128]},
                    'batch_size': {'values': [32, 64, 128]},
                    'max_epochs_mean': {'value': max_epochs_mean},
                }
            }
            sweep_id = wandb.sweep(sweep_config, project=project_name_mean)
            def _fit_network_mean_wandb():
                wandb.init()
                self.fit_network_mean(wandb.config, save_model=False)
            # count = number of runs in sweep for random
            wandb.agent(sweep_id, function=_fit_network_mean_wandb, count=count)
            wandb.finish()
            
        elif self.run_mode_mean == 'best':
            wandb.login()
            api = wandb.Api()
            # get sweep name
            #sweeps = api.sweeps("kstoreyf-stanford-university", project_name_mean)
            sweeps = api.project(project_name_mean).sweeps()

            self.sweep = next((s for s in sweeps if s.name == self.sweep_name_mean), None)
            assert self.sweep is not None, f"Sweep {self.sweep_name_mean} not found"
            #self.sweep = api.sweep(f"kstoreyf-stanford-university/{project_name_mean}/sweeps/{sweep_id}")
            # Get best run parameters
            best_run = self.sweep.best_run()
            print(f"Using hyperparameters from best run from sweep {self.sweep_name_mean}: {best_run.config}")
            wandb.init(project=project_name_mean, config=best_run.config)
            self.fit_network_mean(wandb.config, save_model=True)
            wandb.finish()
            
        elif self.run_mode_mean == 'single':
            # do not use 'parameters' keyword here, flat dict! 
            # that's the issue if get AttributeError that the parameter names aren't recognized
            config = {
                'learning_rate': {'value': 1e-3},
                'hidden_size': {'value': 64},
                'batch_size': {'value': 32},
                'max_epochs_mean': {'value': max_epochs_mean},
            }
            wandb.init(project=project_name_mean, config=config)
            self.fit_network_mean(wandb.config, save_model=True)
            wandb.finish()
            
        elif self.run_mode_mean == 'load':
            self.load_model_mean()
            self.n_params = self.model_mean.output_shape[-1]
            print("n_params from saved model:", self.n_params)
        
        elif self.run_mode_mean is None:
            print("self.run_mode_mean is None, continuing")
            pass
        
        else:
            raise ValueError(f'Invalid run_mode_mean: {self.run_mode_mean}')
        
        ### Cov model ###

        project_name_cov = 'muchisimocks-mn-cov'
        
        if self.run_mode_cov == 'sweep':
            self.setup_scaler_cov()
            count = 10 #if random
            sweep_config = {
                'name': self.sweep_name_cov,
                #'method': 'grid',
                'method': 'random',
                'metric': {
                    'name': 'val_loss',
                    'goal': 'minimize'
                },
                'parameters': {
                    'learning_rate': {'values': [1e-2, 1e-3, 1e-4, 1e-5]},
                    'hidden_size': {'values': [32, 64, 128]},
                    'batch_size': {'values': [32, 64, 128]},
                    'max_epochs_cov': {'value': max_epochs_cov},
                }
            }
            sweep_id = wandb.sweep(sweep_config, project=project_name_cov)
            def _fit_network_cov_wandb():
                wandb.init()
                self.fit_network_covariances(wandb.config, save_model=False)
            # count = number of runs in sweep for random
            wandb.agent(sweep_id, function=_fit_network_cov_wandb, count=count)
        
        elif self.run_mode_cov == 'best':
            print("Running run_mode_cov == 'best'")
            self.setup_scaler_cov()
            wandb.login()
            api = wandb.Api()
            # get sweep name
            sweeps = api.project(project_name_cov).sweeps()

            self.sweep = next((s for s in sweeps if s.name == self.sweep_name_cov), None)
            assert self.sweep is not None, f"Sweep {self.sweep_name_cov} not found"
            # Get best run parameters
            best_run = self.sweep.best_run()
            print(f"Using hyperparameters from best run from sweep {self.sweep_name_cov}: {best_run.config}")
            wandb.init(project=project_name_cov, config=best_run.config)
            self.fit_network_covariances(wandb.config, save_model=True)
            #self.fit_network_covariances(best_run.config, save_model=True)
            
        elif self.run_mode_cov == 'single':
            self.setup_scaler_cov()
            # do not use 'parameters' keyword here, flat dict! 
            # that's the issue if get AttributeError that the parameter names aren't recognized
            config = {
                'learning_rate': {'value': 1e-2},
                'hidden_size': {'value': 64},
                'batch_size': {'value': 32},
                'max_epochs_cov': {'value': max_epochs_cov},
            }
            wandb.init(project=project_name_cov, config=config)
            self.fit_network_covariances(wandb.config, save_model=True)
            
        elif self.run_mode_cov == 'load':
            self.load_model_cov()
            self.load_scaler_cov()
            fn_cov_dict = f'{self.dir_mn}/cov_dict.npy'
            self.cov_dict = np.load(fn_cov_dict, allow_pickle=True).item()
        
        elif self.run_mode_cov is None:
            print("self.run_mode_cov is None, continuing")
            pass
        
        else:
            raise ValueError(f'Invalid run_mode_cov: {self.run_mode_cov}')

        # if self.run_mode_cov is not None:
        #     self.setup_scaler_cov()
        
        # if self.run_mode_cov == 'single':
        #     self.fit_network_covariances(max_epochs=max_epochs_cov)
        # elif self.run_mode_cov is None:
        #     print("self.run_mode_cov is None, continuing")
        #     pass
        # else:
        #     raise ValueError(f'Invalid run_mode_mean: {self.run_mode_mean}')


    def setup_scaler_cov(self):
        theta_pred_train = self.model_mean.predict(self.y_train)
        covariances_train = self.get_covariances(self.theta_train, theta_pred_train,
                                                 include_covariances=self.include_covariances)
        #self.scaler_cov = scl.Scaler(func='none')
        self.scaler_cov = scl.Scaler(func='minmax')
        #self.scaler_cov = scl.Scaler(func='log_minmax_const')
        #self.scaler_cov = scl.Scaler(func='log_minmax')
        self.scaler_cov.fit(covariances_train)
        fn_scaler_cov = f'{self.dir_mn}/scaler_cov.p'
        # need pickle for custom object!!
        with open(fn_scaler_cov, "wb") as f:
            pickle.dump(self.scaler_cov, f)
        #np.save(fn_scaler_cov, self.scaler_cov)
        
        
    def load_scaler_cov(self):
        fn_scaler_cov = f'{self.dir_mn}/scaler_cov.p'
        #self.scaler_cov = np.load(fn_scaler_cov, allow_pickle=True)
        with open(fn_scaler_cov, "rb") as f:
            self.scaler_cov = pickle.load(f)
        
    
    def setup_scaler_y(self):
        self.scaler_y = scl.Scaler()
        self.scaler_y.fit(self.y_train_unscaled)
        
        self.y_train = self.scaler_y.scale(self.y_train_unscaled)
        self.y_val = self.scaler_y.scale(self.y_val_unscaled)
        if self.y_test_unscaled is not None:
            self.y_test = self.scaler_y.scale(self.y_test_unscaled)
        
        fn_scaler_y = f'{self.dir_mn}/scaler_y.p'
        
        # need pickle for custom object!!
        with open(fn_scaler_y, "wb") as f:
            pickle.dump(self.scaler_y, f)


    def load_scaler_y(self):
        fn_scaler_y = f'{self.dir_mn}/scaler_y.p'
        #self.scaler_cov = np.load(fn_scaler_cov, allow_pickle=True)
        with open(fn_scaler_y, "rb") as f:
            self.scaler_y = pickle.load(f)
        
    
    def load_model_mean(self):
        assert os.path.exists(f'{self.dir_mn}/model_mean.keras'), f"model_mean.keras not found in {self.dir_mn}"
        print(f"Loading mean model from {self.dir_mn}")
        self.model_mean = tf.keras.models.load_model(f'{self.dir_mn}/model_mean.keras')


    def load_model_cov(self):
        assert os.path.exists(f'{self.dir_mn}/model_cov.keras'), f"model_cov.keras not found in {self.dir_mn}"
        print(f"Loading covariance model from {self.dir_mn}")
        self.model_cov = tf.keras.models.load_model(f'{self.dir_mn}/model_cov.keras')


        
    def fit_network_mean(self, config, save_model=True):

        #wandb.init()
        
        print("wandb.config:", config)

        # Set up mean model
        print("n_dim:", self.n_dim)
        #if len(self.n_dim) == 1:
        if isinstance(self.n_dim, int):
            model_instance = nn.NeuralNetwork(self.n_dim, self.n_params, 
                                        hidden_size=config.hidden_size,
                                        learning_rate=config.learning_rate) 
        elif len(self.n_dim) == 3:
            model_instance = nn.ConvolutionalNeuralNetwork(self.n_dim, self.n_params, 
                                        hidden_size=config.hidden_size,
                                        learning_rate=config.learning_rate)
        else:
            raise ValueError("Other dimensions not implemented!")
        self.model_mean = model_instance.model() 
        
        # Train mean model
        start_from_epoch = 200
        if config.max_epochs_mean < start_from_epoch:
            start_from_epoch = 0
            
        callback_earlystop = tf.keras.callbacks.EarlyStopping(patience=25,
                                    restore_best_weights=True,
                                    start_from_epoch=start_from_epoch
        )
        
        callback_wandb = wandb.keras.WandbCallback()
        
        hist = self.model_mean.fit(self.y_train, self.theta_train,
                                epochs=config.max_epochs_mean, batch_size=config.batch_size, shuffle=True,
                                callbacks=[callback_earlystop, callback_wandb],
                                validation_data=(self.y_val, self.theta_val))
        
        # if a sweep has a single run, let's save the model
        if save_model:
            np.save(f'{self.dir_mn}/model_mean_history.npy', hist.history)
            self.model_mean.save(f'{self.dir_mn}/model_mean.keras')
        


    def fit_network_covariances(self, config, save_model=True):
        print("Covariance config:", config)

        theta_pred_train = self.model_mean.predict(self.y_train)
        theta_pred_val = self.model_mean.predict(self.y_val)
        covariances_train = self.get_covariances(self.theta_train, theta_pred_train,
                                                 include_covariances=self.include_covariances)
        covariances_val = self.get_covariances(self.theta_val, theta_pred_val,
                                               include_covariances=self.include_covariances)
        
        covariances_train_scaled = self.scaler_cov.scale(covariances_train)
        covariances_val_scaled = self.scaler_cov.scale(covariances_val)
        
        print('cov unscaled min max:', np.min(covariances_train, axis=0), np.max(covariances_train, axis=0))
        print('cov scaled min max:', np.min(covariances_train_scaled, axis=0), np.max(covariances_train_scaled, axis=0))
        
        print('cov dec unscaled:', covariances_train[0])
        print('cov dec scaled:', covariances_train_scaled[0])
        #print(sdfsd)
        
        # Set up covariance network
        model_instance = nn.NeuralNetwork(self.n_dim, self.n_covs, 
                                    hidden_size=config.hidden_size,
                                    learning_rate=config.learning_rate,
                                    activation='leakyrelu',
                                    alpha=0.1)
        self.model_cov = model_instance.model()
        
        start_from_epoch = 200
        if config.max_epochs_cov < start_from_epoch:
            start_from_epoch = 0
        callback_earlystop = tf.keras.callbacks.EarlyStopping(patience=25,
                                    restore_best_weights=True,
                                    start_from_epoch=start_from_epoch)
        
        callback_wandb = wandb.keras.WandbCallback()

        hist = self.model_cov.fit(self.y_train,
                                    covariances_train_scaled,
                                    epochs=config.max_epochs_cov, batch_size=config.batch_size, shuffle=True,
                                    callbacks=[callback_earlystop, callback_wandb],
                                    validation_data = (self.y_val,
                                                        covariances_val_scaled))
        
        if save_model:
            np.save(f'{self.dir_mn}/model_cov_history.npy', hist.history)
            self.model_cov.save(f'{self.dir_mn}/model_cov.keras')

        
    def get_covariances_orig(self, theta_true, theta_pred, include_covariances=True):
        
        covariances = []
        self.cov_dict = {}
        count = 0
        for i in range(self.n_params):
            for j in range(self.n_params):
                if not include_covariances:
                    if i!=j:
                        continue
                    
                if j<i:
                    self.cov_dict[(i,j)] = self.cov_dict[(j,i)]
                    continue
                    
                covariances.append((theta_true[:,i]-theta_pred[:,i])* \
                                   (theta_true[:,j]-theta_pred[:,j]))
                
                self.cov_dict[(i,j)] = count
                count += 1
                
        covariances = np.array(covariances).T

        self.n_covs = covariances.shape[1]
        return covariances
    

    def get_covariances(self, theta_true, theta_pred, include_covariances=True):
        
        covmats = []
        self.cov_dict = {}
        #count = 0
        #print('cov')
        #print(theta_true.shape)
        
        covmats = np.zeros((theta_true.shape[0], self.n_params, self.n_params))
        for i in range(self.n_params):
            for j in range(self.n_params):
                if not include_covariances:
                    if i!=j:
                        continue
                
                covmats[:,i,j] = (theta_true[:,i]-theta_pred[:,i])* \
                                 (theta_true[:,j]-theta_pred[:,j])    
        
        covdatas = []
        first = True
        for covmat in covmats:
            # to ensure PSD, regularize first
            covdata, cov_dict = self.covmat_to_covdata(covmat, include_covariances=include_covariances,
                                                       cov_mode=self.cov_mode)
            if first:
                # cov_dicts all the save, just store one
                self.cov_dict = cov_dict
                print(cov_dict)
                first = False
            covdatas.append(covdata)
        covdatas = np.array(covdatas)
            
        fn_cov_dict = f'{self.dir_mn}/cov_dict.npy'
        np.save(fn_cov_dict, self.cov_dict)
        #print(covariances_decomposed_unrolled.shape)

        self.n_covs = covdatas.shape[1]
        return covdatas


    def covmat_to_covdata(self, covmat, include_covariances=True,
                          cov_mode='direct'):
        if cov_mode=='cholesky':
            # Cholesky decomposition to ensure PSD
            epsilon = 1e-15
            cov_psd = covmat + np.identity(self.n_params) * epsilon
            cov_decomposed = np.linalg.cholesky(cov_psd)
        
            # unroll
            covdata, cov_dict = self.unroll_covariance_matrix(cov_decomposed,
                                            include_covariances=include_covariances)
        elif cov_mode=='eigs':
            eigenvalues, Q_eigenvectors = np.linalg.eigh(covmat)
            Q_eigenvectors_unrolled, cov_dict = self.unroll_covariance_matrix(Q_eigenvectors,
                                            include_covariances=include_covariances)
            covdata = np.hstack((eigenvalues, Q_eigenvectors_unrolled))
            # this is how we'd get the rotated covmat, but don't need
            C_rotated = np.dot(np.dot(Q_eigenvectors.T, covmat), Q_eigenvectors)
            print(eigenvalues)
            print(C_rotated)
        elif cov_mode=='direct':
            covdata, cov_dict = self.unroll_covariance_matrix(covmat,
                                            include_covariances=include_covariances)
        else:
            raise ValueError(f"cov_mode {cov_mode} not recognized")
               
        return covdata, cov_dict


    def covdata_to_covmat(self, covdata, cov_dict, include_covariances=True,
                          cov_mode='direct'):
        if cov_mode=='cholesky':
            # reroll
            cov_decomposed = self.reroll_covariance_matrix(covdata, cov_dict, include_covariances=include_covariances)
            # rebuild matrix after cholesky decomposition
            covmat = cov_decomposed @ cov_decomposed.T
        elif cov_mode=='eigs':
            eigenvalues, Q_eigenvectors_unrolled = covdata[:self.n_params], covdata[self.n_params:]
            Q_eigenvectors = self.reroll_covariance_matrix(Q_eigenvectors_unrolled, cov_dict, include_covariances=include_covariances)
            Lambda = np.diag(eigenvalues)
            covmat = np.dot(np.dot(Q_eigenvectors, Lambda), Q_eigenvectors.T)
        elif cov_mode=='direct':
            covmat = self.reroll_covariance_matrix(covdata, cov_dict, include_covariances=include_covariances)
        else:
            raise ValueError(f"cov_mode {cov_mode} not recognized")

        return covmat


    def unroll_covariance_matrix(self, covmat, include_covariances=True):
        covmat_unrolled = []
        count = 0
        cov_dict = {}
        for i in range(self.n_params):
            for j in range(self.n_params):
                if not include_covariances:
                    if i!=j:
                        continue
                cov_dict[(i,j)] = count
                if j<=i:
                    covmat_unrolled.append(covmat[i,j])
                    count += 1   
        return covmat_unrolled, cov_dict
            

    def reroll_covariance_matrix(self, covmat_unrolled, cov_dict, include_covariances=True):
        covmat = np.zeros((self.n_params, self.n_params))
        for i in range(self.n_params):
            for j in range(self.n_params):
                if not include_covariances:
                    if i!=j:
                        continue
                covmat[i,j] = covmat_unrolled[cov_dict[(i,j)]]
        return covmat


    def evaluate_orig(self, y_obs, mean_only=False):

        theta_pred = self.model_mean.predict(np.atleast_2d(y_obs))

        if mean_only: 
            return theta_pred

        covariances_pred_scaled = self.model_cov.predict(np.atleast_2d(y_obs))
        covariances_pred = self.scaler_cov.unscale(covariances_pred_scaled)

        covs_pred = []
        for covariance_pred in covariances_pred:
            cov_pred = np.zeros((self.n_params, self.n_params))
            for i in range(self.n_params):
                for j in range(self.n_params):
                    if not self.include_covariances:
                        if i!=j:
                            continue
                    cov_pred[i,j] = covariance_pred[self.cov_dict[(i,j)]]
            covs_pred.append(cov_pred)    

        # samples_test = np.array(np.random.multivariate_normal(theta_pred[0], cov_pred,int(1e6)),dtype=np.float32)            
        return theta_pred, covs_pred


    def evaluate(self, y_obs_unscaled, mean_only=False):
        print("Evaluating")
        y_obs = self.scaler_y.scale(y_obs_unscaled)
        
        theta_pred = self.model_mean.predict(np.atleast_2d(y_obs))

        if mean_only: 
            return theta_pred

        covdata_pred_scaled = self.model_cov.predict(np.atleast_2d(y_obs))
        covdata_pred = self.scaler_cov.unscale(covdata_pred_scaled)

        covs_pred = []
        for cov_pred in covdata_pred:
            covmat = self.covdata_to_covmat(cov_pred, self.cov_dict, include_covariances=self.include_covariances,
                                            cov_mode=self.cov_mode)
            covs_pred.append(covmat)
 
        return theta_pred, covs_pred


    def evaluate_test_set(self, y_test_unscaled=None, theta_test=None, tag_test=''):
        
        if y_test_unscaled is None:
            y_test_unscaled = self.y_test_unscaled
        theta_test_pred, covs_test_pred = self.evaluate(y_test_unscaled)
        fn_theta_test_pred = f'{self.dir_mn}/theta_test{tag_test}_pred.npy'
        fn_covs_test_pred = f'{self.dir_mn}/covs_test{tag_test}_pred.npy'
        np.save(fn_theta_test_pred, theta_test_pred)
        np.save(fn_covs_test_pred, covs_test_pred)
        
        # if we have a test set, let's also compute and save the covariances
        # for help with dev
        if theta_test is None:
            theta_test = self.theta_test
        if theta_test is not None:
            covdata_test = self.get_covariances(theta_test, theta_test_pred,
                                    include_covariances=self.include_covariances)
            covs_test = []
            for cov_test in covdata_test:
                covmat = self.covdata_to_covmat(cov_test, self.cov_dict, include_covariances=self.include_covariances,
                                                cov_mode=self.cov_mode)
                covs_test.append(covmat)
            np.save(f'{self.dir_mn}/covs_test{tag_test}.npy', covs_test)




