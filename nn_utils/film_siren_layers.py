import tensorflow as tf


class Sine(tf.keras.layers.Layer):
    def __init__(self, w0: float = 30.0, **kwargs):
        """
        Sine activation function with w0 scaling support.
        Args:
            w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`
        """
        super(Sine, self).__init__(**kwargs)
        self.w0 = w0

    def call(self, inputs):
        return tf.sin(self.w0 * inputs)

    def get_config(self):
        config = {"w0": self.w0}
        base_config = super(Sine, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FiLMSiren(tf.keras.layers.Layer):
    def __init__(self, w0=30.0, **kwargs):
        super(FiLMSiren, self).__init__(trainable=False, **kwargs)
        self.sine = Sine(w0=w0)

    @tf.function
    def call(self, tensor, frequency, phase_shift):
        return self.sine(tensor * frequency + phase_shift)
