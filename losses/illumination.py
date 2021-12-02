import tensorflow as tf


def mean_absolute_logarithmic_error(y_true, y_pred):
    """Computes the mean absolute logarithmic error between `y_true` and `y_pred`.
    `loss = mean(abs(log(y_true + 1) - log(y_pred + 1)), axis=-1)`
    Standalone usage:
    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = mean_absolute_logarithmic_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> y_true = np.maximum(y_true, 1e-7)
    >>> y_pred = np.maximum(y_pred, 1e-7)
    >>> assert np.allclose(
    ...     loss.numpy(),
    ...     np.mean(
    ...         np.abs(np.log(y_true + 1.) - np.log(y_pred + 1.)), axis=-1))
    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    Returns:
      Mean absolute logarithmic error values. shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    first_log = tf.math.log(tf.maximum(y_pred, 1e-7) + 1.0)
    second_log = tf.math.log(tf.maximum(y_true, 1e-7) + 1.0)
    return tf.reduce_mean(tf.math.abs(first_log - second_log), axis=-1)


class MeanAbsoluteLogarithmicError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return mean_absolute_logarithmic_error(y_true, y_pred)


def mean_squared_logarithmic_error(y_true, y_pred):
    """Computes the mean squared logarithmic error between `y_true` and `y_pred`.
    `loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)`
    Standalone usage:
    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> y_true = np.maximum(y_true, 1e-7)
    >>> y_pred = np.maximum(y_pred, 1e-7)
    >>> assert np.allclose(
    ...     loss.numpy(),
    ...     np.mean(
    ...         np.square(np.log(y_true + 1.) - np.log(y_pred + 1.)), axis=-1))
    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    Returns:
      Mean squared logarithmic error values. shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    first_log = tf.math.log(tf.maximum(y_pred, 1e-7) + 1.0)
    second_log = tf.math.log(tf.maximum(y_true, 1e-7) + 1.0)
    return tf.reduce_mean(tf.math.square(first_log - second_log), axis=-1)


class MeanSquareLogarithmicError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return mean_squared_logarithmic_error(y_true, y_pred)
