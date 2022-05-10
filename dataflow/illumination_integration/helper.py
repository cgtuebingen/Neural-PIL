import os
import re
from glob import glob
from typing import List

import cv2
import numpy as np
import pyexr
import tensorflow as tf
import tensorflow_addons as tfa

import nn_utils.math_utils as math_utils


def read_image(path):
    _, extension = os.path.splitext(path)
    is_hdr = extension == ".hdr"
    is_exr = extension == ".exr"

    if is_hdr:
        img = cv2.cvtColor(
            cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB
        ).astype(np.float32)
    elif is_exr:
        img = pyexr.read(path)
    else:
        raise Exception("Environment maps need to be .exr or .hdr files.")

    if img.min() < 0:
        img = img + img.min()

    return np.clip(np.nan_to_num(img, nan=0, posinf=np.max(img), neginf=0), 0, None)


def getBilinearFromUv(env_map: tf.Tensor, uvs: tf.Tensor) -> tf.Tensor:
    u = uvs[..., 0]
    v = uvs[..., 1]

    # vFlipped = 1 - v

    # u corresponds to width and v to heights
    u_reshaped = tf.reshape(u, (-1, tf.math.reduce_prod(u.shape[1:])))
    v_reshaped = tf.reshape(v, (-1, tf.math.reduce_prod(v.shape[1:])))

    uvs = tf.stack(
        [u_reshaped * (env_map.shape[2] - 1), v_reshaped * (env_map.shape[1] - 1)],
        axis=-1,
    )

    return tfa.image.interpolate_bilinear(env_map, uvs, indexing="xy")


def getBilinearFromDirections(env_map: tf.Tensor, directions: tf.Tensor) -> tf.Tensor:
    uvs = math_utils.direction_to_uv(directions)
    return getBilinearFromUv(env_map, uvs)


def get_mip_level(path):
    return int(re.search("(?<=mip)\d", os.path.basename(path)).group(0))


def get_all_env_map_paths(hdridir, env_name):
    return sorted(
        glob(os.path.join(hdridir, env_name + "_mip*.exr",)), key=get_mip_level,
    )


def load_data(hdridir) -> List[np.ndarray]:
    base_files = os.path.join(hdridir, "*_mip0.exr")

    all_files = sorted(glob(base_files), key=os.path.basename)
    env_names = [os.path.basename(p).replace("_mip0.exr", "") for p in all_files]
    num_levels = len(get_all_env_map_paths(hdridir, env_names[0]))

    mips = []
    for i in range(num_levels):
        mip_level = [
            read_image(os.path.join(hdridir, p + "_mip%d.exr" % i))[..., :3]
            for p in env_names
        ]
        mips.append(np.stack(mip_level))

    return mips


def split_dataset(dataset: List[np.ndarray], val_examples: int = 30):
    train_samples = [d[val_examples:] for d in dataset]
    val_samples = [d[:val_examples] for d in dataset]

    len_train = train_samples[0].shape[0]

    return train_samples, val_samples, len_train
