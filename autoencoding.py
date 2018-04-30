import keras
from keras import backend as K
from keras.layers import Dense, Input, Lambda
from keras.models import Model


def binarize(x):
    return x + K.stop_gradient(K.round(x) - x)
Binarize = Lambda(binarize, output_shape=lambda x: x, name='encoding')


class AutoEncoder(Model):
    """Autoencoder

    Wrapper around several `keras.models.Model`s exposing methods for
    training an autoencoder and then encoding and decoding input vectors.
    """
    def __init__(self, input_dim, latent_dim, intermediate_dims, output_activation):
        """Initialize an ``AutoEncoder``.

        Args:
            input_dim (int): Dimension of the input.
            latent_dim (int): Dimension of the "latent representation" or
            intermediate_dims (list): List of `int`s representing the
                dimmension of the hidden layers up to, but not including, the
                latent layer. See the example below.

        Example
        -------
        The instance

        >>> autoencoder = AutoEncoder(784, 32, [256, 128])

        will have the following architecture ::
            
            |--------- 784 ---------|       INPUT

               |------ 256 ------|

                  |--- 128 ---|

                    |-  32 -|               CODE

                  |--- 128 ---|

               |------ 256 ------|

            |--------- 784 ---------|       OUTPUT


        Usage
        -----
        >>> autoencoder.fit(x_train, validation_data=x_test)
        >>> encodings = autoencoder.encode(x_test)
        >>> decodings = autoencoder.decode(encodings)
        """
        self._encoder = self._decoder = self._model = None
        model_input, model_output = self._init_model(
            input_dim, latent_dim, intermediate_dims, output_activation)
        super().__init__(inputs=model_input, outputs=model_output)


    def _init_model(self, input_dim, latent_dim, intermediate_dims, output_activation):
        """Create ``self._model`` for training the autoencoder as well as
        ``self._encoder`` and ``self._decoder`` for encoding/decoding
        output/codes.
        """
        model_input = Input(shape=(input_dim,))
        model_output = model_input
        for dim in intermediate_dims:
            model_output = Dense(dim, activation='relu')(model_output)
        model_output = Dense(latent_dim, activation='sigmoid')(model_output)
        model_output = Binarize(model_output)

        for dim in reversed(intermediate_dims):
            model_output = Dense(dim, activation='relu')(model_output)
        model_output = Dense(input_dim, activation=output_activation)(model_output)

        return model_input, model_output

    @property
    def encoder(self):
        return Model(
            inputs=model.input,
            outputs=model.get_layer('encoding').output)
