import os
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import nn_utils.math_utils as math_utils
import utils.training_setup_utils as train_utils
from dataflow.brdf_ae import init_dataset
from models.brdf_ae_net import BrdfInterpolatingAutoEncoder
from nn_utils.tensorboard_visualization import to_8b


def validation_run(model, test_data, no_roughness=False):
    mseMean = tf.keras.metrics.Mean()

    if no_roughness:
        test_data = test_data[..., :6]

    (gt, pred, batch_mse) = model.test_step(test_data)

    mseMean.update_state(batch_mse)

    newShape = (
        1,
        int(np.sqrt(test_data.shape[0])),
        int(np.sqrt(test_data.shape[0])),
        6 if args.no_roughness else 7,
    )

    gt = tf.reshape(gt, newShape)
    pred = tf.reshape(pred, newShape)
    spacer = tf.ones_like(gt[:, :1])
    joined = tf.concat([gt, spacer, pred], 1)

    tf.summary.image("val/diffuse", to_8b(joined[..., :3]))
    tf.summary.image("val/specular", to_8b(joined[..., 3:6]))
    if not no_roughness:
        tf.summary.image("val/roughness", to_8b(joined[..., 6:]))

    mse = mseMean.result()
    tf.summary.scalar("val_loss", mse)

    return mse


def main(args):
    with train_utils.SetupDirectory(args, copy_files=True, main_script=__file__):
        assert train_utils.get_num_gpus() <= 1

        datapath = os.path.join(args.datadir, "brdf_parameters.npy")
        testpath = os.path.join(args.testdir, "brdf_parameters.npy")
        no_roughness = args.no_roughness

        global_batch_size = args.batch_size * train_utils.get_num_gpus()
        train_dataset, test_dataset = init_dataset(
            datapath, testpath, global_batch_size
        )

        epoch_length = 200000
        stats_every = epoch_length // 20
        imgs_every = epoch_length // 8

        # Optimizer and models
        model = BrdfInterpolatingAutoEncoder(args)

        start_step = model.restore()
        tf.summary.experimental.set_step(start_step)

        start_epoch = start_step // len(train_dataset)

        print(
            "Starting training in epoch {} at step {}".format(start_epoch, start_step)
        )

        # Initial validation run
        validation_run(model, test_dataset, no_roughness)

        # Run training
        brdf_dims = 6 if no_roughness else 7
        for epoch in range(start_epoch + 1, args.epochs + 1):
            start_time = time.time()
            cur_step = 0
            with tqdm(total=epoch_length) as pbar:
                for x in train_dataset:
                    (x_recon, interpolated_samples, z, losses,) = model.train_step(x,)

                    if tf.summary.experimental.get_step() % stats_every == 0:

                        tf.summary.histogram("Embedding", z)

                        for k, v in losses.items():
                            tf.summary.scalar(k, v)

                    if tf.summary.experimental.get_step() % imgs_every == 0:
                        newShape = (
                            1,
                            int(np.sqrt(global_batch_size)),
                            int(np.sqrt(global_batch_size)),
                            6 if no_roughness else 7,
                        )

                        gt = tf.reshape(
                            x[: global_batch_size ** 2, :brdf_dims], newShape
                        )
                        recon = tf.reshape(x_recon[: global_batch_size ** 2], newShape)

                        spacer = tf.ones_like(gt[:, :1])
                        joined = tf.concat([gt, spacer, recon], 1)

                        # Reconstruction quality
                        tf.summary.image("train/diffuse", to_8b(joined[..., :3]))
                        tf.summary.image("train/specular", to_8b(joined[..., 3:6]))
                        if not no_roughness:
                            tf.summary.image("train/roughness", to_8b(joined[..., 6:]))

                        ismpl = tf.reshape(
                            interpolated_samples,
                            (
                                1,
                                -1,
                                args.interpolation_samples,
                                6 if args.no_roughness else 7,
                            ),
                        )

                        without_roughness = ismpl[..., :6]
                        if not no_roughness:
                            roughness_3ch = math_utils.repeat(ismpl[..., 6:], 3, -1)
                            ismpl = tf.concat([without_roughness, roughness_3ch], -1)

                        ismplStacked = tf.reshape(
                            ismpl,
                            (1, ismpl.shape[1], args.interpolation_samples, -1, 3),
                        )
                        ismplWidthStacked = tf.concat(
                            [
                                ismplStacked[..., i, :]
                                for i in range(ismplStacked.shape[-2])
                            ],
                            2,
                        )

                        tf.summary.image("interpol/joined", to_8b(ismplWidthStacked))

                        # Random newly generated
                        mean = tf.math.reduce_mean(z)
                        stddev = tf.math.reduce_std(z)
                        random_gen = model.decoder.random_sample(
                            global_batch_size, mean, stddev
                        )

                        random_gen = tf.reshape(random_gen, newShape)
                        tf.summary.image("sample/diffuse", to_8b(random_gen[..., :3]))
                        tf.summary.image("sample/specular", to_8b(random_gen[..., 3:6]))
                        if not no_roughness:
                            tf.summary.image(
                                "sample/roughness", to_8b(random_gen[..., 6:])
                            )

                    tf.summary.experimental.set_step(
                        tf.summary.experimental.get_step() + 1
                    )
                    if tf.summary.experimental.get_step() % 100 == 0:
                        pbar.update(100)
                        pbar.set_postfix(
                            loss=losses["loss"].numpy(),
                            reconstruction=losses["reconstruction_loss"].numpy(),
                            smoothness=losses["smoothness_loss"].numpy(),
                        )

                    cur_step += 1
                    if cur_step % epoch_length == 0:
                        break

            model.save(
                tf.summary.experimental.get_step()
            )  # Step was already incremented

            end_time = time.time()

            mse = validation_run(model, test_dataset)
            print(
                "Epoch: {}, Test set MSE: {}, time elapse for current epoch: {}".format(
                    epoch, mse, end_time - start_time
                )
            )


def parser():
    parser = train_utils.setup_parser()

    parser.add_argument("--datadir", required=True, type=str)
    parser.add_argument("--testdir", required=True, type=str)

    parser.add_argument("--images_per_batch", type=int, default=16)
    parser.add_argument("--net_w", type=int, default=32)
    parser.add_argument("--net_d", type=int, default=2)
    parser.add_argument("--disc_w", type=int, default=32)
    parser.add_argument("--disc_d", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=3)

    parser.add_argument("--fourier_res", default=10, type=int)

    parser.add_argument("--interpolation_samples", default=8, type=int)
    parser.add_argument("--lambda_generator_loss", type=float, default=1e-2)
    parser.add_argument("--lambda_cyclic_loss", type=float, default=1e-2)
    parser.add_argument("--lambda_smoothness_loss", type=float, default=1e-1)
    parser.add_argument("--lambda_distance_loss", type=float, default=0)

    parser.add_argument("--no_roughness", action="store_true")

    return parser


if __name__ == "__main__":
    args = parser().parse_args()

    main(args)
