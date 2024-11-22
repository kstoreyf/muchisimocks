import tensorflow as tf
from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.layers import Dense, Dropout, ReLU, LeakyReLU, Input, \
                                    Conv3D, MaxPool3D, GlobalAveragePooling3D, BatchNormalization
import keras


# Following demos at https://github.com/NiallJeffrey/MomentNetworks/tree/master

class NeuralNetwork():
    """
    A simple MLP with LeakyReLU activation
    """
    
    def __init__(self, input_size, output_size, 
                 hidden_size=32, learning_rate=None,
                 activation='leakyrelu',
                 alpha=0.1):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        :param learning_rate: learning rate for the optimizer
        """
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.hidden_size = hidden_size
        if activation=='leakyrelu':
            self.activation_func = LeakyReLU
            self.activation_kwargs = {'alpha': alpha}
        elif activation=='relu':
            self.activation_func = ReLU
            self.activation_kwargs = {}
        else:
            raise ValueError(f"Activation function {activation} not recognized")

        
    def model(self):
        
        input_data = (Input(shape=(self.input_size,)))
        #input_err = (Input(shape=(self.input_size,)))

        x1 = Dense(self.hidden_size, input_dim=self.input_size, kernel_initializer='normal')(input_data)
        x1 = self.activation_func(**self.activation_kwargs)(x1)
        x2 = Dense(self.hidden_size, kernel_initializer='normal')(x1)
        x2 = self.activation_func(**self.activation_kwargs)(x2)
        x3 = Dense(self.hidden_size, kernel_initializer='normal')(x2)
        x3 = self.activation_func(**self.activation_kwargs)(x3)
        #x4 = Dense(self.output_size, kernel_initializer='normal', activation='relu')(x3)        
        x4 = Dense(self.output_size, kernel_initializer='normal')(x3)        

        dense_model = Model(input_data, x4)
        #dense_model = Model([input_data, input_err], x4)
        dense_model.summary()
        
        # # this is not what we want bc the losses are in terms of thetas!! 
        # def loss_wrapper(y_true, y_pred):
        #     def loss_gaussian_nll(y_true, y_pred, y_err, epsilon=1e-8):
        #         # epsilon for numerical stability; see 
        #         y_var = keras.backend.square(y_err) + epsilon  # Convert stddev to variance
        #         return keras.backend.log(y_err + epsilon) + ((y_true - y_pred) ** 2) / (2 * y_var)
        #     return loss_gaussian_nll(y_true, y_pred, input_err)

        if self.learning_rate is None:
            dense_model.compile(optimizer='adam', 
                                loss='mse',
                                #loss=loss_wrapper
            )
        else:
            dense_model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), 
                                loss='mse',
                                #loss=loss_wrapper
            ) 

        return dense_model
    
    
class ConvolutionalNeuralNetwork():
    
    def __init__(self, input_size, output_size, 
                 hidden_size=32, learning_rate=None,
                 #activation='leakyrelu',
                 alpha=0.1):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.hidden_size = hidden_size
        # if activation=='leakyrelu':
        #     self.activation_func = LeakyReLU
        #     self.activation_kwargs = {'alpha': alpha}
        # elif activation=='relu':
        #     self.activation_func = ReLU
        #     self.activation_kwargs = {}
        # else:
        #     raise ValueError(f"Activation function {activation} not recognized")

        
    def model(self):
        print(self.input_size)
        
        input_data = (Input(shape=(*self.input_size, 1)))
        print(input_data.shape, self.output_size)
        print("yo")
        
        x = Conv3D(filters=64, kernel_size=3, activation="relu")(input_data)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)
        
        x = GlobalAveragePooling3D()(x)
        x = Dense(units=512, activation="relu")(x)
        
        # via https://keras.io/examples/vision/3D_image_classification/
        # x = Conv3D(filters=64, kernel_size=3, activation="relu")(input_data)
        # x = MaxPool3D(pool_size=2)(x)
        # x = BatchNormalization()(x)
        # print("made it here")
        # x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
        # x = MaxPool3D(pool_size=2)(x)
        # x = BatchNormalization()(x)

        # x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
        # x = MaxPool3D(pool_size=2)(x)
        # x = BatchNormalization()(x)

        # x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
        # x = MaxPool3D(pool_size=2)(x)
        # x = BatchNormalization()(x)

        # x = GlobalAveragePooling3D()(x)
        # x = Dense(units=512, activation="relu")(x)
        # x = Dropout(0.3)(x)

        #outputs = Dense(units=1, activation="sigmoid")(x) # this was for classification
        outputs = Dense(self.output_size, kernel_initializer='normal')(x) # copying from Pk model for regression

        # Define the model.
        model = Model(input_data, outputs, name="CNN3D")
        model.summary()
        
        if self.learning_rate is None:
            model.compile(optimizer='adam', loss='mse')
        else:
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss='mse')

        return model

   