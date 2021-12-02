import tensorflow as tf
from tensorflow.python.keras.engine.input_layer import InputLayer

from nn_utils.nerf_layers import FourierEmbedding


class BrdfEncoder(tf.keras.Model):
    def __init__(
        self,
        args,
        input_dimensions: int = 7,
        activation="relu",
        final_activation="linear",
        kernel_initializer="he_uniform",
        bias_initializer=tf.keras.initializers.Zeros,
        **kwargs,
    ):
        super(BrdfEncoder, self).__init__(**kwargs)

        net = [tf.keras.Input((input_dimensions,))]
        for _ in range(args.net_d):
            net.append(
                tf.keras.layers.Dense(
                    args.net_w,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                )
            )
        net.append(
            tf.keras.layers.Dense(
                args.latent_dim,
                activation=final_activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )
        )

        self.net = tf.keras.Sequential(net)

    @tf.function
    def call(self, x):
        x = self.net(x)

        return x


class BrdfDecoder(tf.keras.Model):
    def __init__(
        self,
        args,
        output_dimensions: int = 7,
        fourier_frequency_embedding: int = 10,
        activation="relu",
        final_activation="sigmoid",
        kernel_initializer="he_uniform",
        bias_initializer=tf.keras.initializers.Zeros,
        **kwargs,
    ):
        super(BrdfDecoder, self).__init__(**kwargs)

        self.latent_dim = args.latent_dim

        embedder = FourierEmbedding(
            fourier_frequency_embedding, input_dim=args.latent_dim
        )
        print(args.latent_dim, embedder.get_output_dimensionality(), args.net_w)

        net = [tf.keras.layers.InputLayer(input_shape=(args.latent_dim,))]
        if fourier_frequency_embedding >= 1:
            net.append(embedder)
            net.append(
                tf.keras.layers.Lambda(
                    lambda x: tf.ensure_shape(
                        x, [None, embedder.get_output_dimensionality()]
                    )
                )
            )

        for _ in range(args.net_d):
            net.append(
                tf.keras.layers.Dense(
                    args.net_w,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                )
            )

        print(args.net_w)

        net.append(
            tf.keras.layers.Dense(
                output_dimensions,
                activation=final_activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )
        )

        print(output_dimensions)

        self.net = tf.keras.Sequential(net)
        print("Done")

    @tf.function
    def call(self, x):
        x = self.net(x)

        return x

    @tf.function
    def random_sample(self, samples, mean=0.0, stddev=0.3):
        x = tf.clip_by_value(
            tf.random.normal(
                shape=(samples, self.latent_dim),
                mean=mean,
                stddev=stddev,
                dtype=tf.float32,
            ),
            -1,
            1,
        )
        return self.call(x)


class BrdfDiscriminator(tf.keras.Model):
    def __init__(
        self,
        args,
        input_dimensions: int = 7,
        activation="relu",
        final_activation="linear",
        kernel_initializer="he_uniform",
        bias_initializer=tf.keras.initializers.Zeros,
        **kwargs,
    ):
        super(BrdfDiscriminator, self).__init__(**kwargs)

        net = [tf.keras.Input((input_dimensions,))]
        for _ in range(args.disc_d):
            net.append(
                tf.keras.layers.Dense(
                    args.disc_w,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                )
            )

        net.append(
            tf.keras.layers.Dense(
                1,
                activation=final_activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )
        )

        self.net = tf.keras.Sequential(net)

    @tf.function
    def call(self, x):
        x = self.net(x)

        return x
