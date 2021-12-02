import tensorflow as tf

import nn_utils.math_utils as math_utils


class DistanceLoss(tf.keras.losses.Loss):
    def euclidean_distance_all_pairs(self, x):
        x0 = tf.expand_dims(x, 0)
        x1 = tf.expand_dims(x, 1)
        return tf.sqrt(tf.reduce_sum(tf.square(x0 - x1), axis=-1) + 1e-7)

    def call(self, y_true, y_pred):
        # Calculate the maximum distance possible in euclidean space 0 to 1
        y_true_dims = y_true.shape[-1]
        y_pred_dims = y_pred.shape[-1]

        max_y_true_dist = tf.sqrt(tf.cast(y_true_dims, tf.float32))
        max_y_pred_dist = tf.sqrt(tf.cast(y_pred_dims, tf.float32))

        y_true_distances = self.euclidean_distance_all_pairs(y_true)
        y_pred_distances = self.euclidean_distance_all_pairs(y_pred * 0.5 + 0.5)

        # Normalize the distances
        norm_y_true = y_true_distances / max_y_true_dist
        norm_y_pred = y_pred_distances / max_y_pred_dist

        distances = norm_y_true - norm_y_pred
        mask = tf.ones_like(y_true_distances) - tf.linalg.diag(
            tf.ones_like(y_true[:, 0])
        )
        return tf.reduce_mean(tf.nn.relu(distances * mask))


def lsganGeneratorLoss(fake_logits):
    return 0.5 * tf.reduce_mean(tf.square(fake_logits - 1))


class LsganDiscriminatorLoss(tf.keras.losses.Loss):
    def call(self, real_logits, fake_logits):
        pos = tf.reduce_mean(tf.square(real_logits - 1))  # Move to 1
        neg = tf.reduce_mean(tf.square(fake_logits))  # Move to 0
        return 0.5 * pos + 0.5 * neg


class CyclicEmbeddingLoss(tf.keras.losses.Loss):
    def call(self, initial_embedding, cyclic_embedding):
        return tf.reduce_mean(math_utils.l2Norm(initial_embedding - cyclic_embedding))


def smoothnessLoss(x):
    return tf.reduce_mean(tf.square(x))
