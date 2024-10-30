import tensorflow as tf
from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.layers import Dense, Dropout, ReLU, LeakyReLU, Input, \
                                    Conv3D, MaxPool3D, GlobalAveragePooling3D, BatchNormalization



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
        print(self.input_size)
        
        input_data = (Input(shape=(self.input_size,)))

        x1 = Dense(self.hidden_size, input_dim=self.input_size, kernel_initializer='normal')(input_data)
        x1 = self.activation_func(**self.activation_kwargs)(x1)
        x2 = Dense(self.hidden_size, kernel_initializer='normal')(x1)
        x2 = self.activation_func(**self.activation_kwargs)(x2)
        x3 = Dense(self.hidden_size, kernel_initializer='normal')(x2)
        x3 = self.activation_func(**self.activation_kwargs)(x3)
        #x4 = Dense(self.output_size, kernel_initializer='normal', activation='relu')(x3)        
        x4 = Dense(self.output_size, kernel_initializer='normal')(x3)        

        dense_model = Model(input_data, x4)
        dense_model.summary()

        if self.learning_rate is None:
            dense_model.compile(optimizer='adam', loss='mse')
        else:
            dense_model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss='mse')

        return dense_model
    
    
class ConvolutionalNeuralNetwork():
    """
    A simple MLP with LeakyReLU activation
    """
    
    def __init__(self, input_size, output_size, 
                 hidden_size=32, learning_rate=None,
                 #activation='leakyrelu',
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
        
        input_data = (Input(shape=(self.input_size, self.input_size, self.input_size, 1)))

        x = Conv3D(filters=64, kernel_size=3, activation="relu")(input_data)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)

        x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)

        x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)

        x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)

        x = GlobalAveragePooling3D()(x)
        x = Dense(units=512, activation="relu")(x)
        x = Dropout(0.3)(x)

        outputs = Dense(units=1, activation="sigmoid")(x)

        # Define the model.
        model.summary()
        model = Model(input_data, outputs, name="CNN3D")
        
        return model

   