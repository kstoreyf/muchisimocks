class Scaler:

    def __init__(self, func='log_minmax'):
        self.func = func
        
    def fit(self, x_train):
        self.x_train_min = np.min(x_train, axis=0)
        self.x_train_max = np.max(x_train, axis=0)

    def scale(self, x):
        if self.func=='log_minmax':
            return self.scale_log_minmax(x)
        elif self.func=='minmax':
            return self.scale_minmax(x)
        else:
            raise ValueError(f"Function {self.func} not recognized")

    def unscale(self, x):
        if self.func=='log_minmax':
            return self.unscale_log_minmax(x)
        elif self.func=='minmax':
            return self.unscale_minmax(x)
        else:
            raise ValueError(f"Function {self.func} not recognized")
       
    def scale_error(self, err, x):
        if self.func=='log_minmax':
            return self.scale_error_log_minmax(err, x)
        elif self.func=='minmax':
            raise ValueError(f"Error not implemented for {self.func}")
        else:
            raise ValueError(f"Function {self.func} not recognized")

    def scale_log_minmax(self, x):
        log_x = np.log10(x)
        log_x_norm = (log_x - np.log10(self.x_train_min)) / (np.log10(self.x_train_max) - np.log10(self.    x_train_min))
        return log_x_norm
    
    def unscale_log_minmax(self, x_scaled):
        x = x_scaled * (np.log10(self.x_train_max) - np.log10(self.x_train_min)) + np.log10(self.x_train_min)
        return 10**x  
        
    def scale_error_log_minmax(self, err, x):
        # need 1/np.log(10) factor bc working in base 10
        dydx = 1./x * 1/np.log(10) * 1./(np.log10(self.x_train_max) - np.log10(self.x_train_min))
        err_scaled = np.sqrt(np.multiply(dydx**2, err**2))
        return err_scaled
    
    def scale_log(self, x):
        return np.log10(x)
    
    def scale_log_error(self, err, x):
        return (1./x) * (1/np.log(10)) * err
    
    def scale_minmax(self, x):
        return (x - self.x_train_min) / (self.x_train_max - self.x_train_min)
    
    def unscale_minmax(self, x_scaled):
        return x_scaled * (self.x_train_max - self.x_train_min) + self.x_train_min

    