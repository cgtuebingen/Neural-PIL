import tensorflow as tf


def slightly_padded_tanh(x):
    return tf.nn.tanh(x) * 1.01


class EmbeddingStore(tf.keras.layers.Embedding):
    def __init__(
        self, num_samples: int, latent_dim: int, latent_mean, latent_std, **kwargs
    ) -> None:
        super(EmbeddingStore, self).__init__(
            num_samples,
            latent_dim,
            embeddings_initializer="zeros",
            input_length=1,
            **kwargs
        )

        self.num_samples = num_samples
        self.latent_dim = latent_dim

        self.latent_mean = tf.convert_to_tensor(latent_mean, tf.float32)
        self.latent_std = tf.convert_to_tensor(latent_std, tf.float32)

    @tf.function
    def convert_noise_to_latent(self, noise):
        tf.debugging.assert_shapes(
            [(noise, ("N", self.latent_dim)),]
        )
        return (
            slightly_padded_tanh(noise * (self.latent_std[None, :] * 3))
            + self.latent_mean[None, :]
        )

    @tf.function
    def call(self, idxs):
        if self.num_samples == 1:
            return self.convert_noise_to_latent(super().call(tf.convert_to_tensor([0])))
        else:
            return self.convert_noise_to_latent(super().call(idxs))
