import numpy as np
import tensorflow as tf


def ensure_input_values_valid(dataset):
    if dataset.dtype == np.uint8:
        dataset = dataset / 255.0
    dataset = dataset.astype(np.float32)
    diffuse = dataset[..., :3]
    specular = dataset[..., 3:6]
    roughness = dataset[..., 6:7]

    is_metal = np.any(specular >= (155 / 255), -1)

    # Diffuse
    diffuse = np.where(
        np.broadcast_to(is_metal[..., np.newaxis], diffuse.shape),
        diffuse,  # if is metal allow diffuse to become 0
        np.where(
            np.broadcast_to(np.mean(diffuse, -1, keepdims=True), diffuse.shape)
            < 30 / 255,
            (30 / 255)
            / np.clip(np.mean(diffuse, -1, keepdims=True), 1 / 255, 1)
            * diffuse,
            diffuse,
        ),  # Else scale it to coal level (30/255)
    )
    # Always scale to (240/255)
    diffuse = np.where(
        np.broadcast_to(np.mean(diffuse, -1, keepdims=True), diffuse.shape) > 240 / 255,
        (240 / 255)
        / np.clip(np.mean(diffuse, -1, keepdims=True), 1 / 255, 1)
        * diffuse,
        diffuse,
    )

    # Specular
    # If material is not metal clip to (40/255) to (75/255)
    # Else allow 155/255 to 255/255

    # ensure lower bounds
    specular = np.where(
        np.broadcast_to(is_metal[..., np.newaxis], diffuse.shape),
        np.where(
            np.broadcast_to(np.mean(specular, -1, keepdims=True), specular.shape)
            < 155 / 255,
            (155 / 255)
            / np.clip(np.mean(specular, -1, keepdims=True), 1 / 255, 1)
            * specular,
            specular,
        ),  # If material is metal ensure that it is at least 155
        np.where(
            np.broadcast_to(np.mean(specular, -1, keepdims=True), specular.shape)
            < 40 / 255,
            (40 / 255)
            / np.clip(np.mean(specular, -1, keepdims=True), 1 / 255, 1)
            * specular,
            specular,
        ),  # if it is not then at least 40
    )

    # ensure upper bounds
    specular = np.where(
        np.broadcast_to(is_metal[..., np.newaxis], diffuse.shape),
        specular,  # No upper limits for metals
        np.where(
            np.broadcast_to(np.mean(specular, -1, keepdims=True), specular.shape)
            > 75 / 255,
            (75 / 255)
            / np.clip(np.mean(specular, -1, keepdims=True), 1 / 255, 1)
            * specular,
            specular,
        ),  # Ensure upper limit for non metals
    )

    return np.concatenate([diffuse, specular, roughness], -1)


def init_dataset(train_path, test_path, batch_size):
    train_data = np.load(train_path)
    test_data = np.load(test_path)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_data)
        .shuffle(int(1e5), reshuffle_each_iteration=True)
        .batch(batch_size)
        .map(lambda x: tf.numpy_function(ensure_input_values_valid, [x], tf.float32))
        .prefetch(1000)
    )
    test_dataset = tf.convert_to_tensor(test_data[:4096], dtype=tf.float32)

    return train_dataset, test_dataset
