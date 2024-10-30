import numpy as np
import pathlib
import tensorflow as tf

import neural_network as nn
import scaler_custom as scl




class MomentNetwork(): 

    def __init__(self, theta_train=None, y_train=None, y_err_train=None,
                       theta_val=None, y_val=None, y_err_val=None,
                       theta_test=None, y_test=None, y_err_test=None,
                       tag_mn='', n_threads=8, cov_mode='direct',
                       ):
        
        if n_threads is not None:
            tf.config.threading.set_inter_op_parallelism_threads(n_threads)
            tf.config.threading.set_intra_op_parallelism_threads(n_threads)

        self.theta_train = theta_train
        self.y_train = y_train
        self.y_err_train = y_err_train

        self.theta_val = theta_val
        self.y_val = y_val
        self.y_err_val = y_err_val

        self.theta_test = theta_test
        self.y_test = y_test
        self.y_err_test = y_err_test
        
        if self.y_train is not None:
            self.n_dim = self.y_train.shape[1]
        elif self.y_test is not None:
            self.n_dim = self.y_test.shape[1]
        if self.theta_train is not None:
            self.n_params = self.theta_train.shape[1]
        elif self.theta_test is not None:
            self.n_params = self.theta_test.shape[1]
        
        self.cov_mode = cov_mode
        self.include_covariances = True
        
        self.dir_mn = f'../results/results_moment_network/mn{tag_mn}'
        p = pathlib.Path(self.dir_mn)
        p.mkdir(parents=True, exist_ok=True)


    def run(self, max_epochs_mean=1500, max_epochs_cov=1000):
        
        self.fit_network_mean(max_epochs=max_epochs_mean)
        self.setup_scaler()
        self.fit_network_covariances(max_epochs=max_epochs_cov)


    def setup_scaler(self):
        theta_pred_train = self.model_mean.predict(self.y_train)
        covariances_train = self.get_covariances(self.theta_train, theta_pred_train,
                                                 include_covariances=self.include_covariances)
        #self.scaler_cov = scl.Scaler(func='none')
        self.scaler_cov = scl.Scaler(func='minmax')
        #self.scaler_cov = scl.Scaler(func='log_minmax_const')
        #self.scaler_cov = scl.Scaler(func='log_minmax')
        self.scaler_cov.fit(covariances_train)
        fn_scaler_cov = f'{self.dir_mn}/scaler_cov.npy'
        np.save(fn_scaler_cov, self.scaler_cov)
        
        
    def load_scaler(self):
        fn_scaler_cov = f'{self.dir_mn}/scaler_cov.npy'
        self.scaler_cov = np.save(fn_scaler_cov)
        
    
    def load_model_mean(self):
        self.model_mean = tf.keras.models.load_model(f'{self.dir_mn}/model_mean.keras')


    def load_model_cov(self):
        self.model_cov = tf.keras.models.load_model(f'{self.dir_mn}/model_cov.keras')

        
        
    def fit_network_mean(self, max_epochs=1500):
        
        # Set up mean model
        model_instance = nn.NeuralNetwork(self.n_dim, self.n_params, 
                                    hidden_size=64,
                                    learning_rate=1e-3) 
        self.model_mean = model_instance.model() 
        
        # Train mean model
        start_from_epoch = 200
        if max_epochs < start_from_epoch:
            start_from_epoch = 0
        callback = tf.keras.callbacks.EarlyStopping(patience=25,
                                    restore_best_weights=True,
                                    start_from_epoch=start_from_epoch)
        
        hist = self.model_mean.fit(self.y_train, self.theta_train,
                                epochs=max_epochs, batch_size=64, shuffle=True,
                                callbacks=[callback],
                                validation_data=(self.y_val, self.theta_val))
        
        np.save(f'{self.dir_mn}/model_mean_history.npy', hist.history)
        self.model_mean.save(f'{self.dir_mn}/model_mean.keras')


    def fit_network_covariances(self, max_epochs=1000):

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
                                    hidden_size=64,
                                    learning_rate=1e-2,
                                    activation='leakyrelu',
                                    alpha=0.1)
        self.model_cov = model_instance.model()
        
        start_from_epoch = 200
        if max_epochs < start_from_epoch:
            start_from_epoch = 0
        callback = tf.keras.callbacks.EarlyStopping(patience=25,
                                    restore_best_weights=True,
                                    start_from_epoch=start_from_epoch)
        
        hist = self.model_cov.fit(self.y_train,
                                    covariances_train_scaled,
                                    epochs=max_epochs, batch_size=64, shuffle=True,
                                    callbacks=[callback],
                                    validation_data = (self.y_val,
                                                        covariances_val_scaled))
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


    def evaluate(self, y_obs, mean_only=False):
        print("Evaluating")
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


    def evaluate_test_set(self):
        
        theta_test_pred, covs_test_pred = self.evaluate(self.y_test)
        np.save(f'{self.dir_mn}/theta_test_pred.npy', theta_test_pred)
        np.save(f'{self.dir_mn}/covs_test_pred.npy', covs_test_pred)

        # if we have a test set, let's also compute and save the covariances
        # for help with dev
        if self.theta_test is not None:
            covdata_test = self.get_covariances(self.theta_test, theta_test_pred,
                                    include_covariances=self.include_covariances)
            covs_test = []
            for cov_test in covdata_test:
                covmat = self.covdata_to_covmat(cov_test, self.cov_dict, include_covariances=self.include_covariances,
                                                cov_mode=self.cov_mode)
                covs_test.append(covmat)
            np.save(f'{self.dir_mn}/covs_test.npy', covs_test)
