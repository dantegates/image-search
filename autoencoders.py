import keras
import keras.backend as K
from keras.layers import Dense, Input, Lambda, Dot


ENCODING_LAYER_NAME = 'encoding'


def binarize(x):
    return x + K.stop_gradient(K.round(x) - x)
Binarize = Lambda(binarize, output_shape=lambda x: x, name=ENCODING_LAYER_NAME)


class AutoEncoder(keras.models.Model):
    """Autoencoder

    Wrapper around several `keras.models.Model`s exposing methods for
    training an autoencoder and then encoding and decoding input vectors.
    """
    def __init__(self, input_dim, latent_dim, intermediate_dims, output_activation):
        """Initialize an ``AutoEncoder``.

        Args:
            input_dim (int): Dimension of the input.
            latent_dim (int): Dimension of the "latent representation" or code.
            intermediate_dims (list): List of `int`s representing the
                dimmension of the hidden layers up to, but not including, the
                latent layer. See the example below.
            output_activation (str or object): The activation used on the final
                output of the autoencoder. This gets passed on to the underlying
                `keras` implementation.

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
        >>> autoencoder = AutoEncoder(...)
        >>> autoencoder.compile(...)
        >>> autoencoder.fit(X_train, X_train, ...)
        >>> encoder = autoencoder.integer_encoder(X)
        ... np.array([[12387],
                      [3982909],
                      ...])
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
    def bit_encoder(self):
        encoding = self.get_layer(ENCODING_LAYER_NAME).get_output_at(-1)
        return keras.models.Model(inputs=self.input, outputs=encoding)

    @property
    def integer_encoder(self):
        encoding = self.get_layer(ENCODING_LAYER_NAME).get_output_at(-1)
        # multiply each bit by its corresponding power of 2 to get d-bit int
        def to_int(X):
            X = K.cast(X, 'int32')  # this only works for latent dims <= 32
            latent_dim = K.int_shape(X)[-1]
            Z = 2**K.arange(latent_dim)
            return K.dot(X, K.reshape(Z, (-1, 1)))
        encoding = Lambda(to_int, output_shape=lambda x: x)(encoding)
        return keras.models.Model(inputs=self.input, outputs=encoding)
