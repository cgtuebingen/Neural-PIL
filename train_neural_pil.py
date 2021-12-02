import os
from typing import Callable, List, Dict

import imageio
import pyexr
import tensorflow as tf
import numpy as np
from tqdm import tqdm

import dataflow.nerd as data
import nn_utils.math_utils as math_utils
import utils.training_setup_utils as train_utils
from models.neural_pil import NeuralPILModel
from nn_utils.nerf_layers import get_full_image_eval_grid
from nn_utils.tensorboard_visualization import hdr_to_tb, horizontal_image_log, to_8b


def add_args(parser):
    parser.add_argument(
        "--log_step",
        type=int,
        default=100,
        help="frequency of tensorboard metric logging",
    )
    parser.add_argument(
        "--weights_epoch", type=int, default=10, help="save weights every x epochs"
    )
    parser.add_argument(
        "--validation_epoch",
        type=int,
        default=5,
        help="render validation every x epochs",
    )
    parser.add_argument(
        "--testset_epoch",
        type=int,
        default=100,
        help="render testset every x epochs",
    )
    parser.add_argument(
        "--video_epoch",
        type=int,
        default=400,
        help="render video every x epochs",
    )

    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=250,
        help="exponential learning rate decay (in 1000s)",
    )

    parser.add_argument("--render_only", action="store_true")
    parser.add_argument("--only_video", action="store_true")
    parser.add_argument("--video_factor", type=int, default=0)

    return parser


def parse_args():
    parser = add_args(
        data.add_args(
            NeuralPILModel.add_args(
                train_utils.setup_parser(),
            ),
        ),
    )
    return train_utils.parse_args_file_without_nones(parser)


def eval_datasets(
    strategy,
    df,
    model,
    hwf,
    near,
    far,
    illum_optimizer,
    steps: int,
    chunk_size: int,
    is_single_env: bool,
    illumination_factor,
):
    # Build lists to save all individual images
    gt_rgbs = []
    gt_masks = []

    H, W, _ = hwf

    predictions = {}
    to_extract_coarse = [("rgb", 3), ("acc_alpha", 1)]
    to_extract_fine = [
        ("rgb", 3),
        ("acc_alpha", 1),
        ("diffuse", 3),
        ("specular", 3),
        ("roughness", 1),
        ("normal", 3),
        ("depth", 1),
    ]

    # Go over validation dataset
    with strategy.scope():
        for dp in tqdm(df):
            img_idx, rays_o, rays_d, pose, mask, ev100, _, _, target = dp

            gt_rgbs.append(tf.reshape(target, (H, W, 3)))
            gt_masks.append(tf.reshape(mask, (H, W, 1)))

            # Optimize SGs first - only if we have varying illumination
            if not is_single_env:
                illum_loss = model.illumination_steps(
                    rays_o,
                    rays_d,
                    pose,
                    near,
                    far,
                    img_idx,
                    ev100,
                    illumination_factor,
                    illum_optimizer,
                    target,
                    steps,
                    chunk_size,
                    strategy,
                )
                print(
                    "Illumination estimation done. Remaining error:", illum_loss.numpy()
                )

            # Render image.
            coarse_result, fine_result = model.distributed_call(
                strategy,
                chunk_size,
                rays_o,
                rays_d,
                pose,
                near,
                far,
                img_idx,
                ev100,
                illumination_factor,
                training=False,
                high_quality=True,
            )

            # Extract values and reshape them to the image dimensions
            new_shape: Callable[[int], List[int]] = lambda x: [H, W, x]

            for name, channels in to_extract_coarse:
                predictions["coarse_%s" % name] = predictions.get(
                    "coarse_%s" % name, []
                ) + [tf.reshape(coarse_result[name], new_shape(channels))]

            for name, channels in to_extract_fine:
                if name in fine_result:
                    predictions["fine_%s" % name] = predictions.get(
                        "fine_%s" % name, []
                    ) + [tf.reshape(fine_result[name], new_shape(channels))]

            # Build the env_map:
            if train_utils.get_num_gpus() > 1:
                # Only a single env map is used
                img_idx = img_idx[:1]
            illumination_context = model.illumination_embedding_store(img_idx)
            env_map = model.fine_model.illumination_net.eval_env_map(
                illumination_context, 0
            )
            predictions["fine_env_map"] = predictions.get("fine_env_map", []) + [
                env_map[0]
            ]

    # Stack all images in dataset in batch dimension
    ret = {}
    ret["gt_rgb"] = tf.stack(gt_rgbs, 0)
    ret["gt_mask"] = tf.stack(gt_masks, 0)

    for pname, vals in predictions.items():
        ret[pname] = tf.stack(vals, 0)

    fine_ssim = tf.reduce_mean(
        tf.image.ssim(
            math_utils.white_background_compose(ret["gt_rgb"], ret["gt_mask"]),
            math_utils.white_background_compose(ret["fine_rgb"], ret["fine_acc_alpha"]),
            max_val=1.0,
        )
    )
    fine_psnr = tf.reduce_mean(
        tf.image.psnr(
            math_utils.white_background_compose(ret["gt_rgb"], ret["gt_mask"]),
            math_utils.white_background_compose(ret["fine_rgb"], ret["fine_acc_alpha"]),
            max_val=1.0,
        )
    )

    return ret, fine_ssim, fine_psnr


def run_validation(
    strategy,
    val_df,
    model,
    hwf,
    near,
    far,
    illum_optimizer,
    chunk_size: int,
    is_single_env: bool,
    illumination_factor,
):

    ret, fine_ssim, fine_psnr = eval_datasets(
        strategy,
        val_df,
        model,
        hwf,
        near,
        far,
        illum_optimizer,
        20,
        chunk_size,
        is_single_env,
        illumination_factor,
    )

    # Log validation dataset
    horizontal_image_log("val/coarse_rgb", ret["gt_rgb"], ret["coarse_rgb"])
    horizontal_image_log("val/fine_rgb", ret["gt_rgb"], ret["fine_rgb"])

    horizontal_image_log("val/coarse_alpha", ret["gt_mask"], ret["coarse_acc_alpha"])
    horizontal_image_log("val/fine_alpha", ret["gt_mask"], ret["fine_acc_alpha"])

    for n, t in ret.items():
        filters = ["rgb", "acc_alpha"]
        if "fine" in n and not any(f in n for f in filters):
            if "normal" in n:
                tf.summary.image("val/" + n, t * 0.5 + 0.5)
            elif "env_map" in n:
                hdr_to_tb("val/env_map", t)
            else:
                tf.summary.image("val/" + n, t)

    tf.summary.scalar("val/ssim", fine_ssim)
    tf.summary.scalar("val/psnr", fine_psnr)


def main(args):
    # Setup directories, logging etc.
    with train_utils.SetupDirectory(
        args,
        copy_files=not args.render_only or not args.only_video,
        main_script=__file__,
        copy_data=["data/neural_pil", "data/illumination"],
    ):
        strategy = (
            tf.distribute.get_strategy()
            if train_utils.get_num_gpus() <= 1
            else tf.distribute.MirroredStrategy()
        )

        # Setup dataflow
        (
            hwf,
            near,
            far,
            render_poses,
            num_images,
            mean_ev100,
            train_df,
            val_df,
            test_df,
        ) = data.create_dataflow(args)

        # Optimizer and models
        with strategy.scope():
            # Setup models
            neuralpil = NeuralPILModel(num_images, args)
            lrate = train_utils.adjust_learning_rate_to_replica(args)
            if args.lrate_decay > 0:
                lrate = tf.keras.optimizers.schedules.ExponentialDecay(
                    lrate, decay_steps=args.lrate_decay * 1000, decay_rate=0.1
                )
            optimizer = tf.keras.optimizers.Adam(lrate)

            illumination_optimizer = tf.keras.optimizers.Adam(1e-2)

        neuralpil.call(
            ray_origins=tf.zeros((1, 3), tf.float32),
            ray_directions=tf.zeros((1, 3), tf.float32),
            camera_pose=tf.eye(3, dtype=tf.float32)[None, ...],
            near_bound=near,
            far_bound=far,
            illumination_idx=tf.zeros((1,), tf.int32),
            ev100=tf.zeros((1,), tf.float32),
            illumination_factor=tf.zeros((1,), tf.float32),
            training=False,
        )

        # Restore if possible
        start_step = neuralpil.restore()
        tf.summary.experimental.set_step(start_step)

        train_dist_df = strategy.experimental_distribute_dataset(train_df)

        start_epoch = start_step // len(train_df)

        print(
            "Starting training in epoch {} at step {}".format(start_epoch, start_step)
        )

        advanced_loss_decay_steps = (
            args.advanced_loss_done // 1
        )  # Will be 1 magnitude lower after advanced_loss_done steps
        advanced_loss_lambda = tf.Variable(1.0, dtype=tf.float32)

        slow_fade_decay_steps = (args.epochs * len(train_df)) // 4
        slow_fade_loss_lambda = tf.Variable(1.0, dtype=tf.float32)
        # Run the actual optimization for x epochs

        illumination_factor = tf.stop_gradient(
            neuralpil.calculate_illumination_factor(
                tf.convert_to_tensor([[0, 1, 0]], tf.float32), mean_ev100
            )
        )

        print("\n\n\tFactor, ev100:", illumination_factor.numpy(), mean_ev100, "\n\n")

        print(
            "Start Rendering..."
            if args.render_only or args.only_video
            else "Start Training..."
        )
        for epoch in range(
            start_epoch + 1,
            args.epochs
            + (
                2 if args.render_only or args.only_video else 1
            ),  # Slight hack to let this loop run when rendering is at the end
        ):
            pbar = tf.keras.utils.Progbar(len(train_df))

            # Iterate over the train dataset
            if not args.render_only and not args.only_video:
                with strategy.scope():
                    for dp in train_dist_df:
                        (
                            img_idx,
                            rays_o,
                            rays_d,
                            pose,
                            mask,
                            ev100,
                            wb,
                            wb_ref_image,
                            target,
                        ) = dp

                        advanced_loss_lambda.assign(
                            1
                            * 0.1
                            ** (
                                tf.summary.experimental.get_step()
                                / advanced_loss_decay_steps
                            )
                        )  # Starts with 1 goes to 0

                        slow_fade_loss_lambda.assign(
                            1
                            * 0.1
                            ** (
                                tf.summary.experimental.get_step()
                                / slow_fade_decay_steps
                            )
                        )  # Starts with 1 goes to 0

                        # Execute train the train step
                        (
                            loss_per_replica,
                            wb_loss_per_replica,
                            coarse_losses_per_replica,
                            fine_losses_per_replica,
                        ) = strategy.run(
                            neuralpil.train_step,
                            (
                                rays_o,
                                rays_d,
                                pose,
                                near,
                                far,
                                img_idx,
                                ev100,
                                illumination_factor,
                                wb_ref_image,
                                wb,
                                optimizer,
                                target,
                                mask,
                                advanced_loss_lambda,
                                slow_fade_loss_lambda,
                            ),
                        )

                        loss = strategy.reduce(
                            tf.distribute.ReduceOp.SUM, loss_per_replica, axis=None
                        )
                        wb_loss = strategy.reduce(
                            tf.distribute.ReduceOp.SUM, wb_loss_per_replica, axis=None
                        )
                        coarse_losses = {}
                        for k, v in coarse_losses_per_replica.items():
                            coarse_losses[k] = strategy.reduce(
                                tf.distribute.ReduceOp.SUM, v, axis=None
                            )
                        fine_losses = {}
                        for k, v in fine_losses_per_replica.items():
                            fine_losses[k] = strategy.reduce(
                                tf.distribute.ReduceOp.SUM, v, axis=None
                            )

                        losses_for_pbar = [
                            ("loss", loss.numpy()),
                            ("coarse_loss", coarse_losses["loss"].numpy()),
                            ("fine_loss", fine_losses["loss"].numpy()),
                        ]

                        pbar.add(
                            1,
                            values=losses_for_pbar,
                        )

                        # Log to tensorboard
                        with tf.summary.record_if(
                            tf.summary.experimental.get_step() % args.log_step == 0
                        ):
                            tf.summary.scalar("loss", loss)
                            tf.summary.scalar("wb_loss", wb_loss)
                            for k, v in coarse_losses.items():
                                tf.summary.scalar("coarse_%s" % k, v)
                            for k, v in fine_losses.items():
                                tf.summary.scalar("fine_%s" % k, v)
                            tf.summary.scalar(
                                "lambda_advanced_loss", advanced_loss_lambda
                            )

                        tf.summary.experimental.set_step(
                            tf.summary.experimental.get_step() + 1
                        )

                # Show last dp and render to tensorboard
                if train_utils.get_num_gpus() > 1:
                    dp = [d.values[0] for d in dp]

                render_test_example(
                    dp, hwf, neuralpil, args, near, far, illumination_factor, strategy
                )

                # Save when a weight epoch arrives
                if epoch % args.weights_epoch == 0:
                    neuralpil.save(
                        tf.summary.experimental.get_step()
                    )  # Step was already incremented

                # Render validation if a validation epoch arrives
                if epoch % args.validation_epoch == 0:
                    print("RENDERING VALIDATION...")
                    # Build lists to save all individual images
                    run_validation(
                        strategy,
                        val_df,
                        neuralpil,
                        hwf,
                        near,
                        far,
                        illumination_optimizer,
                        args.batch_size,
                        args.single_env,
                        illumination_factor,
                    )

            # Render test set when a test epoch arrives
            if epoch % args.testset_epoch == 0 or args.render_only:
                print("RENDERING TESTSET...")
                ret, fine_ssim, fine_psnr = eval_datasets(
                    strategy,
                    test_df,
                    neuralpil,
                    hwf,
                    near,
                    far,
                    illumination_optimizer,
                    100,
                    args.batch_size,
                    args.single_env,
                    illumination_factor,
                )

                if not args.single_env:
                    neuralpil.save(
                        tf.summary.experimental.get_step() + 1
                    )  # Step was already incremented

                testimgdir = os.path.join(
                    args.basedir,
                    args.expname,
                    "test_imgs_{:06d}".format(tf.summary.experimental.get_step()),
                )

                print("Mean PSNR:", fine_psnr, "Mean SSIM:", fine_ssim)
                os.makedirs(testimgdir, exist_ok=True)
                # Save all images in the test_dir
                alpha = ret["fine_acc_alpha"]
                for n, t in ret.items():
                    print(n, t.shape)
                    for b in range(t.shape[0]):
                        to_save = t[b]
                        if "normal" in n:
                            to_save = (t[b] * 0.5 + 0.5) * alpha[b] + (1 - alpha[b])

                        if "env_map" in n:
                            imageio.imwrite(
                                os.path.join(testimgdir, "{:d}_{}.png".format(b, n)),
                                to_8b(
                                    math_utils.linear_to_srgb(to_save / (1 + to_save))
                                )
                                .numpy()
                                .astype(np.uint8),
                            )
                            pyexr.write(
                                os.path.join(testimgdir, "{:d}_{}.exr".format(b, n)),
                                to_save.numpy(),
                            )
                        elif "normal" in n or "depth" in n:
                            pyexr.write(
                                os.path.join(testimgdir, "{:d}_{}.exr".format(b, n)),
                                to_save.numpy(),
                            )
                            if "normal" in n:
                                imageio.imwrite(
                                    os.path.join(
                                        testimgdir, "{:d}_{}.png".format(b, n)
                                    ),
                                    to_8b(to_save).numpy(),
                                )
                        else:
                            imageio.imwrite(
                                os.path.join(testimgdir, "{:d}_{}.png".format(b, n)),
                                to_8b(to_save).numpy().astype(np.uint8),
                            )

            # Render video when a video epoch arrives
            if epoch % args.video_epoch == 0 or args.render_only or args.only_video:
                print("RENDERING VIDEO...")
                video_dir = os.path.join(
                    args.basedir,
                    args.expname,
                    "video_{:06d}".format(tf.summary.experimental.get_step()),
                )
                video_img_dir = os.path.join(
                    video_dir,
                    "images",
                )
                os.makedirs(video_img_dir, exist_ok=True)

                render_video(
                    neuralpil,
                    hwf,
                    test_df,
                    render_poses,
                    strategy,
                    near,
                    args,
                    far,
                    video_img_dir,
                    video_dir,
                    args.video_factor,
                )

            if args.render_only or args.only_video:
                return


def render_video(
    model,
    hwf,
    test_df,
    render_poses,
    strategy,
    near,
    args,
    far,
    video_img_dir,
    video_dir,
    render_factor=0,
):
    H, W, F = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        F = F / render_factor

    fine_results = {}

    for d in test_df:
        img_idx, _, _, _, _, ev100_video, _, _, target = d
        break

    video_latent = model.illumination_embedding_store(img_idx)
    illumination_factor_video = model.calculate_illumination_factor(
        tf.convert_to_tensor([[0, 1, 0]], tf.float32),
        ev100_video,
        video_latent,
    )
    novel_video_latents = tf.convert_to_tensor(
        np.load("data/neural_pil/video_latents.npy"), tf.float32
    )
    random_latents = novel_video_latents
    start_end = video_latent.numpy()
    if len(start_end.shape) != len(random_latents.shape):
        start_end = start_end[None, ...]
    random_latents = tf.concat([start_end, random_latents, start_end], 0)

    pose_df = tf.data.Dataset.from_tensor_slices(render_poses[:, :3, :4])

    # TODO move to render function
    # Render all render_poses
    def render_pose(pose):
        # Always start and end with mean
        rays_o, rays_d = get_full_image_eval_grid(H, W, F, tf.reshape(pose, (3, 4)))

        _, fine_result = model.distributed_call(
            strategy=strategy,
            chunk_size=args.batch_size,
            ray_origins=tf.reshape(rays_o, (-1, 3)),
            ray_directions=tf.reshape(rays_d, (-1, 3)),
            camera_pose=pose,
            near_bound=near,
            far_bound=far,
            illumination_idx=img_idx,
            ev100=ev100_video,
            illumination_factor=illumination_factor_video,
            training=False,
            high_quality=True,
        )

        return fine_result

    for pose_dp in tqdm(pose_df):
        cur_pose = pose_dp
        fine_result = render_pose(pose_dp)

        fine_result["rgb"] = math_utils.white_background_compose(
            math_utils.linear_to_srgb(
                math_utils.uncharted2_filmic(fine_result["hdr_rgb"] * math_utils.ev100_to_exp(ev100_video))
            ),
            fine_result["acc_alpha"][..., None]
            * (
                tf.where(
                    fine_result["depth"] < (far * 0.9),
                    tf.ones_like(fine_result["depth"]),
                    tf.zeros_like(fine_result["depth"]),
                )[..., None]
            ),
        )

        for k, v in fine_result.items():
            fine_results[k] = fine_results.get(k, []) + [v.numpy()]

    # Select random illumination embeddings

    num_illuminations = novel_video_latents.shape[0]
    num_seconds = 4
    num_fps = 30

    # TODO move to function
    total_frames = num_seconds * num_fps
    # Last latent do not require interpolation
    frames_per_illumination = total_frames // (num_illuminations - 1)
    total_frames = frames_per_illumination * (
        num_illuminations - 1
    )  # Make sure that everything fits

    frame_latents = []
    frame_env_idx = 0
    imageio.plugins.freeimage.download()

    env_maps = []

    for latent0, latent1 in zip(random_latents, random_latents[1:]):
        for frame in range(frames_per_illumination):
            blend_alpha = frame / (frames_per_illumination - 1)
            cur_latent = latent0 * (1 - blend_alpha) + latent1 * blend_alpha

            frame_latents.append(cur_latent)

            env_map = model.fine_model.illumination_net.eval_env_map(
                tf.convert_to_tensor(cur_latent[None, ...], dtype=tf.float32),
                float(0),
            )
            env_maps.append(env_map.numpy()[0])

            imageio.imwrite(
                os.path.join(video_img_dir, "env_{:06d}.exr".format(frame_env_idx)),
                env_map.numpy()[0],
            )
            frame_env_idx += 1

    number_of_latent_frames = len(frame_latents)
    # pad frame latents if required
    div_remain = np.ceil(number_of_latent_frames / train_utils.get_num_gpus())
    mod_remain = int(
        (div_remain * train_utils.get_num_gpus()) - number_of_latent_frames
    )
    for _ in range(mod_remain):
        frame_latents.append(frame_latents[-1])  # Clone last

    frame_latents_pad = np.stack(frame_latents)

    latent_df = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(frame_latents_pad, dtype=tf.float32)
    ).batch(train_utils.get_num_gpus())
    latent_dist_df = strategy.experimental_distribute_dataset(latent_df)

    # TODO move to render function
    # Render all latent vectors
    with strategy.scope():
        # Use last pose
        rays_o, rays_d = get_full_image_eval_grid(H, W, F, tf.reshape(cur_pose, (3, 4)))

        print(cur_pose.shape)
        c_pose = cur_pose
        if len(cur_pose.shape) == 3:
            c_pose = cur_pose[0]

        def render_latent(rays_o, rays_d, fres, latents):
            tf.debugging.assert_shapes(
                [
                    (rays_o, ("H", "W", 3)),
                    (rays_d, ("H", "W", 3)),
                    (
                        latents,
                        (
                            1,
                            model.fine_model.illumination_net.latent_units,
                        ),
                    ),
                ]
            )

            view_direction = math_utils.normalize(-1 * tf.reshape(rays_d, (-1, 3)))

            (
                view_direction,
                reflection_direction,
            ) = model.fine_model.renderer.calculate_reflection_direction(
                view_direction,
                fres["normal"],
                camera_pose=c_pose if args.rotating_object else None,
            )

            diffuse_irradiance = model.fine_model.illumination_net.call_multi_samples(
                tf.expand_dims(reflection_direction, 0),
                tf.expand_dims(
                    tf.ones_like(
                        fres["roughness"]
                    ),  # Just sample with maximum roughness
                    0,
                ),
                latents,
            )[0]

            specular_irradiance = model.fine_model.illumination_net.call_multi_samples(
                tf.expand_dims(reflection_direction, 0),
                tf.expand_dims(fres["roughness"], 0),
                latents,
            )[0]

            hdr_rgb = model.fine_model.renderer(
                view_direction,
                fres["normal"],
                diffuse_irradiance,
                specular_irradiance,
                fres["diffuse"],
                fres["specular"],
                fres["roughness"],
            )

            # We do not have a fitting ev for this scene - Just use reinhard tone mapping
            rgb = math_utils.white_background_compose(
                math_utils.linear_to_srgb(math_utils.uncharted2_filmic(hdr_rgb)),
                fres["acc_alpha"][..., None]
                * (
                    tf.where(
                        fres["depth"] < (far * 0.9),
                        tf.ones_like(fres["depth"]),
                        tf.zeros_like(fres["depth"]),
                    )[..., None]
                ),
            )

            return rgb

        for latent_dp in tqdm(latent_dist_df):
            rgb_per_replica = strategy.run(
                render_latent, (rays_o, rays_d, fine_result, latent_dp)
            )
            rgb_result = strategy.gather(rgb_per_replica, 0).numpy()
            rgb_results = np.split(rgb_result, train_utils.get_num_gpus(), 0)
            fine_results["rgb"] = fine_results.get("rgb", []) + rgb_results

    # Everything is now a numpy
    fine_result_np = {
        k: np.stack(v, 0)[: render_poses.shape[0] + number_of_latent_frames]
        for k, v in fine_results.items()
    }
    # reshape and extract
    rgb = fine_result_np["rgb"]
    rgb = rgb.reshape((-1, H, W, 3))

    # save individual images and video
    imageio.mimwrite(
        os.path.join(video_dir, "rgb.mp4"),
        (rgb * 255).astype(np.uint8),
        fps=30,
        quality=8,
    )

    for i in range(rgb.shape[0]):
        imageio.imwrite(
            os.path.join(video_img_dir, "rgb_{:06d}.png".format(i)),
            (rgb[i] * 255).astype(np.uint8),
        )

    alpha = fine_result_np["acc_alpha"].reshape((-1, H, W, 1))
    parameters = {}
    parameters["diffuse"] = math_utils.linear_to_srgb(
        (fine_result_np["diffuse"].reshape((-1, H, W, 3)) * alpha) + (1 - alpha)
    ).numpy()
    parameters["specular"] = math_utils.linear_to_srgb(
        (fine_result_np["specular"].reshape((-1, H, W, 3)) * alpha) + (1 - alpha)
    ).numpy()
    parameters["roughness"] = (
        fine_result_np["roughness"].reshape((-1, H, W, 1)) * alpha
    ) + (1 - alpha)
    parameters["normal"] = math_utils.linear_to_srgb(
        ((fine_result_np["normal"].reshape((-1, H, W, 3)) * 0.5 + 0.5) * alpha)
        + (1 - alpha)
    ).numpy()

    for n, imgs in parameters.items():
        imageio.mimwrite(
            os.path.join(video_dir, "{}.mp4".format(n)),
            (imgs * 255).astype(np.uint8),
            fps=30,
            quality=8,
        )

        for i in range(imgs.shape[0]):
            imageio.imwrite(
                os.path.join(video_img_dir, "{}_{:06d}.png".format(n, i)),
                (imgs[i] * 255).astype(np.uint8),
            )


def render_test_example(dp, hwf, model, args, near, far, illumination_factor, strategy):
    with strategy.scope():
        (
            img_idx,
            _,
            _,
            pose,
            _,
            ev100,
            _,
            _,
            _,
        ) = dp

        H, W, F = hwf
        rays_o, rays_d = get_full_image_eval_grid(H, W, F, pose[0])

        coarse_result, fine_result = model.distributed_call(
            strategy=strategy,
            chunk_size=args.batch_size,
            ray_origins=tf.reshape(rays_o, (-1, 3)),
            ray_directions=tf.reshape(rays_d, (-1, 3)),
            camera_pose=pose,
            near_bound=near,
            far_bound=far,
            illumination_idx=img_idx,
            ev100=ev100,
            illumination_factor=illumination_factor,
            training=False,
            high_quality=True,
        )  # TODO figure out distirbution

        horizontal_image_log(
            "train/rgb",
            tf.reshape(coarse_result["rgb"], (1, H, W, 3)),
            tf.reshape(fine_result["rgb"], (1, H, W, 3)),
        )
        horizontal_image_log(
            "train/alpha",
            tf.reshape(coarse_result["acc_alpha"], (1, H, W, 1)),
            tf.reshape(fine_result["acc_alpha"], (1, H, W, 1)),
        )

        for n, t in fine_result.items():
            filters = ["rgb", "alpha"]
            if not any(f in n for f in filters):
                if "normal" in n:
                    tf.summary.image(
                        "train/" + n, tf.reshape(t * 0.5 + 0.5, (1, H, W, 3))
                    )
                else:
                    if len(t.shape) == 1:
                        t = t[:, None]
                    tf.summary.image(
                        "train/" + n, tf.reshape(t, (1, H, W, t.shape[-1]))
                    )

        env_latent = model.illumination_embedding_store(img_idx)
        env_map = model.fine_model.illumination_net.eval_env_map(env_latent, 0)
        hdr_to_tb("train/env_map", env_map)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    main(args)
