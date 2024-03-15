import numpy as np
import tensorflow as tf

import neural_network as nn
import scaler_custom as scl


n_threads = 24
tf.config.threading.set_inter_op_parallelism_threads(n_threads)
tf.config.threading.set_intra_op_parallelism_threads(n_threads)


class MomentNetwork(): 

    def __init__(self, theta_train, theta_val, theta_test,
                       y_train, y_val, y_test,
                       y_err_train, y_err_val, y_err_test):
        
        self.theta_train = theta_train
        self.theta_val = theta_val
        self.theta_test = theta_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.y_err_train = y_err_train
        self.y_err_val = y_err_val
        self.y_err_test = y_err_test
        
        self.n_dim = self.theta_train.shape[0]
        self.n_params = self.theta_train.shape[1]
        
        self.include_covariances = True


    def run(self):
        
        self.fit_network_mean()
        self.fit_network_variances()

        
        
    def fit_network_mean(self):
        # Set up mean model
        model_instance = nn.NeuralNetwork(self.n_dim, self.n_params, 
                                    hidden_size=64,
                                    learning_rate=1e-3) 
        model_mean = model_instance.model() 
        
        # Train mean model
        callback = tf.keras.callbacks.EarlyStopping(patience=25,
                                    restore_best_weights=True,
                                    start_from_epoch=200)
        history = self.model_mean.fit(self.y_train_scaled, self.theta_train,
                                epochs=2500, batch_size=64, shuffle=True,
                                callbacks=[callback],
                                validation_data=(self.y_val_scaled, self.theta_val))


    def fit_network_variances(self):

        covariances_train = self.get_variances(self, include_covariances=self.include_covariances)
        covariances_val = self.get_variances(self, include_covariances=self.include_covariances)
        
        self.scaler_cov = scl.Scaler(func='minmax')
        self.scaler_cov.fit(covariances_train)
        covariances_train_scaled = self.scaler_cov.scale(covariances_train)
        covariances_val_scaled = self.scaler_cov.scale(covariances_val)
        
        # Set up covariance network
        model_instance = nn.NeuralNetwork(self.n_dim, self.n_covs, 
                                    hidden_size=64,
                                    learning_rate=1e-2,
                                    activation='leakyrelu',
                                    alpha=0.1)
        self.model_var = model_instance.model()
        
        callback = tf.keras.callbacks.EarlyStopping(patience=20,
                                    restore_best_weights=True,
                                    start_from_epoch=100)
        
        history = self.model_var.fit(self.y_train_scaled,
                                                covariances_train_scaled,
                                                epochs=500, batch_size=64, shuffle=True,
                                                callbacks=[callback],
                                                validation_data = (self.y_val_scaled,
                                                                    covariances_val_scaled))

        
    def get_variances(self, include_covariances=True):
        
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
                    
                covariances.append((self.theta_true[:,i]-self.theta_pred[:,i])* \
                                   (self.theta_true[:,j]-self.theta_pred[:,j]))
                
                self.cov_dict[(i,j)] = count
                count += 1
                
        covariances = np.array(covariances).T

        self.n_covs = covariances.shape[1]
        return covariances


    def evaluate(self):
        
        covariances_test = self.get_variances(self, include_covariances=self.include_covariances)
        covariances_test_scaled = self.scaler_cov.scale(covariances_test)

        predicted_mean_obs_test = self.model_mean.predict(np.atleast_2d(self.y_data))

        predicted_var_obs_test_scaled = (self.model_var.predict(np.atleast_2d(self.y_data))[0])
        predicted_var_obs_test = self.scaler_cov.unscale(predicted_var_obs_test_scaled)

        moment_network_param_cov_test = np.zeros((self.n_params, self.n_params))
        for i in range(self.n_params):
            for j in range(self.n_params):
                if not self.include_covariances:
                    if i!=j:
                        continue
                moment_network_param_cov_test[i,j] = predicted_var_obs_test[self.cov_dict[(i,j)]]

        # evaluate mean model
        self.theta_train_pred = self.model_mean.predict(np.atleast_2d(self.y_train_scaled))
        self.theta_val_pred = self.model_mean.predict(np.atleast_2d(self.y_val_scaled))
        self.theta_test_pred = self.model_mean.predict(np.atleast_2d(self.y_test_scaled))

        moment_network_samples_test = np.array(np.random.multivariate_normal(predicted_mean_obs_test[0],
                                            moment_network_param_cov_test,int(1e6)),dtype=np.float32)
    