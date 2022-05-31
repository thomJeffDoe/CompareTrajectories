from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np


class ConvAutoencoder:
    def __init__(self, width, height, depth, filters=(32, 64), latentDim=16, lr=1e-3):
        self.encoder, self.decoder, self.autoencoder = self.build(
            width, height, depth, filters=(32, 64), latentDim=16
        )
        self.compile(lr=lr)

    @staticmethod
    def build(width, height, depth, filters=(32, 64), latentDim=16):
        # initialize the input shape to be "channels last" along with
        # the channels dimension itself

        # ENCODER
        inputShape = (height, width, depth)
        chanDim = -1
        inputs = Input(shape=inputShape)
        x = inputs
        for f in filters:
            x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)
        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim)(x)
        encoder = Model(inputs, latent, name="encoder")

        # DECODER
        latentInputs = Input(shape=(latentDim,))
        x = Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
        # loop over our number of filters again, but this time in
        # reverse order
        for f in filters[::-1]:
            # apply a CONV_TRANSPOSE => RELU => BN operation
            x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)
        x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
        outputs = Activation("sigmoid")(x)
        decoder = Model(latentInputs, outputs, name="decoder")

        # AUTOENCODER
        autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")
        return (encoder, decoder, autoencoder)

    def compile(self, lr):
        self.autoencoder.compile(loss="mse", optimizer=Adam(lr=lr))

    def predict_latent(self,data):
        data = np.array(data).astype("float32") / 255.0
        latent_povs = self.encoder.predict(data, verbose=10)
        return latent_povs
