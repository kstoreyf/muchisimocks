import numpy as np


class Scaler:

    def __init__(self, func):
        assert func in ['none', 'log_minmax', 'log_minmax_const', 'log', 'minmax'], f"Function {func} not recognized"
        self.func = func
        
    def fit(self, x_train):
        self.x_train_min = np.min(x_train, axis=0)
        self.x_train_max = np.max(x_train, axis=0)

    def scale(self, x):
        if self.func=='log_minmax':
            return self._scale_log_minmax(x)
        elif self.func=='log_minmax_const':
            return self._scale_log_minmax_const(x)
        if self.func=='log':
            return self._scale_log(x)
        elif self.func=='minmax':
            return self._scale_minmax(x)
        elif self.func=='none':
            return x
        else:
            raise ValueError(f"Function {self.func} not recognized")

    def unscale(self, x):
        if self.func=='log_minmax':
            return self._unscale_log_minmax(x)
        if self.func=='log_minmax_const':
            return self._unscale_log_minmax_const(x)
        if self.func=='log':
            return self._unscale_log(x)
        elif self.func=='minmax':
            return self._unscale_minmax(x)
        elif self.func=='none':
            return x
        else:
            raise ValueError(f"Function {self.func} not recognized")
       
    def scale_error(self, err, x):
        if self.func=='log_minmax':
            return self._scale_error_log_minmax(err, x)
        if self.func=='log':
            return self._scale_error_log(err, x)
        elif self.func=='none':
            return x
        else:
            raise ValueError(f"Function {self.func} not implemented/recognized")

    def _scale_log_minmax(self, x):
        log_x = np.log10(x)
        log_x_norm = (log_x - np.log10(self.x_train_min)) / (np.log10(self.x_train_max) - np.log10(self.x_train_min))
        return log_x_norm
    
    def _unscale_log_minmax(self, x_scaled):
        x = x_scaled * (np.log10(self.x_train_max) - np.log10(self.x_train_min)) + np.log10(self.x_train_min)
        x = 10**x
        return x  
        
    def _scale_error_log_minmax(self, err, x):
        # need 1/np.log(10) factor bc working in base 10
        dydx = 1./x * 1/np.log(10) * 1./(np.log10(self.x_train_max) - np.log10(self.x_train_min))
        err_scaled = np.sqrt(np.multiply(dydx**2, err**2))
        return err_scaled
    
    def _scale_log_minmax_const(self, x):
        #print('min:', self.x_train_min)
        #print(np.min(x, axis=0))
        x_train_min_const = 10 * np.abs(self.x_train_min) # shift so no negative values, plus a buffer bc other data sets may be diff
        x_train_max_const = self.x_train_max + x_train_min_const
        
        x += x_train_min_const
        #print(np.min(x, axis=0))
        log_x = np.log10(x)
        log_x_norm = (log_x - np.log10(x_train_min_const)) / (np.log10(x_train_max_const) - np.log10(x_train_min_const))
        #print(klsdfs)
        return log_x_norm
    
    def _unscale_log_minmax_const(self, x_scaled):
        x_train_min_const = 10 * np.abs(self.x_train_min)
        x_train_max_const = self.x_train_max + x_train_min_const
        
        x = x_scaled * (np.log10(x_train_max_const) - np.log10(x_train_min_const)) + np.log10(x_train_min_const)
        x = 10**x
        x -= x_train_min_const
        return x    
    
    def _scale_log(self, x):
        return np.log10(x)
    
    def _unscale_log(self, x_scaled):
        x = 10**x_scaled
        return x  
    
    def _scale_error_log(self, err, x):
        return (1./x) * (1/np.log(10)) * err
    
    def _scale_minmax(self, x):
        return (x - self.x_train_min) / (self.x_train_max - self.x_train_min)
    
    def _unscale_minmax(self, x_scaled):
        return x_scaled * (self.x_train_max - self.x_train_min) + self.x_train_min

    