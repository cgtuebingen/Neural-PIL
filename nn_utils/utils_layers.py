import tensorflow as tf


class Interpolation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Interpolation, self).__init__(trainable=False, **kwargs)

    @tf.function
    def call(self, x1, x2, alpha):
        return x1 * tf.cast(1 - alpha, x1.dtype) + x2 * tf.cast(alpha, x2.dtype)

    @tf.function
    def multiple_alphas(self, x1, x2, alphas):
        alphas = tf.reshape(alphas, (-1, *[1 for _ in x1.shape]))
        return x1[None, ...] * tf.cast(1 - alphas, x1.dtype) + x2[None, ...] * tf.cast(
            alphas, x2.dtype
        )
