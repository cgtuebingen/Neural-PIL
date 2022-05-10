import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import tensorflow as tf

import nn_utils.math_utils as math_utils
from losses import multi_gpu_wrapper
from models.neural_pil.embd_store import EmbeddingStore
from models.neural_pil.models import CoarseModel, FineModel
from nn_utils.nerf_layers import add_base_args
from utils.training_setup_utils import (
    StateRestoration,
    StateRestorationItem,
    get_num_gpus,
)


class NeuralPILModel(tf.keras.Model):
    def __init__(self, num_images, args, **kwargs):
        super(NeuralPILModel, self).__init__(**kwargs)

        # Setup the models
        self.fine_model = FineModel(args, **kwargs)
        illumination_latent_dim = self.fine_model.illumination_net.latent_units
        self.coarse_model = CoarseModel(illumination_latent_dim, args, **kwargs)

        self.rotating_object = args.rotating_object

        # Randomize if training
        self.randomized = args.perturb == 1.0
        print("Running with pertubation:", self.randomized)

        self.advanced_loss_done = args.advanced_loss_done

        # Setup the place where the SGs are stored
        self.single_env = args.single_env
        num_illuminations = 1 if args.single_env else num_images
        mean_std = np.load(
            os.path.join(
                args.illumination_network_path, "illumination_latent_mean_std.npy"
            ),
            allow_pickle=True,
        )
        self.illumination_embedding_store = EmbeddingStore(
            num_illuminations,
            illumination_latent_dim,
            latent_mean=mean_std[0],
            latent_std=mean_std[1],
        )
        self.illumination_embedding_store(
            tf.convert_to_tensor([0])
        )  # Ensure the store is built

        # Add loss for wb
        self.num_gpu = max(1, get_num_gpus())
        self.global_batch_size = args.batch_size * self.num_gpu

        self.mse = multi_gpu_wrapper(
            tf.keras.losses.MeanSquaredError, self.global_batch_size,
        )

        self.cosine_similarity = multi_gpu_wrapper(
            tf.keras.losses.CosineSimilarity, self.global_batch_size
        )

        # Setup the state restoration
        states = [
            StateRestorationItem("coarse", self.coarse_model),
            StateRestorationItem("fine", self.fine_model),
            StateRestorationItem("illuminations", self.illumination_embedding_store),
        ]
        self.state_restoration = StateRestoration(args, states)

    def save(self, step):
        # Save weights for step
        self.state_restoration.save(step)

    def restore(self, step: Optional[int] = None) -> int:
        # Restore weights from step or if None the latest one
        return self.state_restoration.restore(step)

    @tf.function
    def call(
        self,
        ray_origins: tf.Tensor,
        ray_directions: tf.Tensor,
        camera_pose: tf.Tensor,
        near_bound: float,
        far_bound: float,
        illumination_idx: tf.Tensor,
        ev100: tf.Tensor,
        illumination_factor: tf.Tensor,
        training=False,
        illumination_context_override=None,
        high_quality=False,
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """Evaluate the network for given ray origins and directions and camera pose

        Args:
            ray_origins (tf.Tensor(float32), [batch, 3]): the ray origin.
            ray_directions (tf.(float32), [batch, 3]): the ray direction.
            camera_pose (tf.Tensor(float32), [batch, 3, 3]): the camera matrix.
            near_bound (float): the near clipping point.
            far_bound (float): the far clipping point.
            illumination_idx (tf.Tensor(int32), [1]): the illumination index.
            ev100 (tf.Tensor(float32), [1]): the ev100 value of the image.
            training (bool, optional): Whether this is a training step or not.
                Activates noise and pertub ray features if requested. Defaults to True.

        Returns:
            coarse_payload (Dict[str, tf.Tensor]): dict with the payload for the coarse
                network.
            fine_payload (Dict[str, tf.Tensor]): dict with the payload for the fine
                network.
        """
        # Get current embedding
        if illumination_context_override is None:
            illumination_context = self.illumination_embedding_store(
                illumination_idx if not self.single_env else tf.convert_to_tensor([0])
            )
        else:
            illumination_context = illumination_context_override

        # Coarse step
        (
            coarse_payload,
            coarse_z_samples,
            coarse_weights,
        ) = self.coarse_model.render_rays(
            ray_origins,
            ray_directions,
            near_bound,
            far_bound,
            tf.stop_gradient(illumination_context),
            randomized=training and self.randomized,
            overwrite_num_samples=(self.coarse_model.num_samples * 2)
            if high_quality
            else None,
        )

        fine_payload, _, _ = self.fine_model.render_rays(
            ray_origins,
            ray_directions,
            coarse_z_samples,
            coarse_weights,
            camera_pose,
            illumination_context,
            ev100,
            illumination_factor,
            randomized=training and self.randomized,
            overwrite_num_samples=(self.fine_model.num_samples * 2)
            if high_quality
            else None,
        )

        return coarse_payload, fine_payload

    def distributed_call(
        self,
        strategy,
        chunk_size: int,
        ray_origins: tf.Tensor,
        ray_directions: tf.Tensor,
        camera_pose: tf.Tensor,
        near_bound: float,
        far_bound: float,
        illumination_idx: tf.Tensor,
        ev100: tf.Tensor,
        illumination_factor: tf.Tensor,
        training=False,
        illumination_context_override=None,
        high_quality=False,
    ):
        if illumination_context_override is not None:
            illumination_idx = tf.cast(
                tf.ones_like(illumination_idx) * illumination_context_override, tf.int32
            )

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        dp_df = (
            tf.data.Dataset.from_tensor_slices((ray_origins, ray_directions))
            .batch(chunk_size // (2 if high_quality else 1) * get_num_gpus())
            .with_options(options)
        )

        dp_dist_df = strategy.experimental_distribute_dataset(dp_df)

        coarse_payloads: Dict[str, List[tf.Tensor]] = {}
        fine_payloads: Dict[str, List[tf.Tensor]] = {}

        def add_to_dict(to_add, main_dict):
            for k, v in to_add.items():
                arr = main_dict.get(k, [],)
                arr.extend(v)
                main_dict[k] = arr

            return main_dict

        for dp in dp_dist_df:
            rays_o, rays_d = dp
            # Render image.
            coarse_result_per_replica, fine_result_per_replica = strategy.run(
                self.call,
                (
                    rays_o,
                    rays_d,
                    camera_pose,
                    near_bound,
                    far_bound,
                    illumination_idx,
                    ev100,
                    illumination_factor,
                    training,
                    illumination_context_override,
                    high_quality,
                ),
            )

            coarse_result = {
                k: strategy.experimental_local_results(v)
                for k, v in coarse_result_per_replica.items()
            }
            fine_result = {
                k: strategy.experimental_local_results(v)
                for k, v in fine_result_per_replica.items()
            }
            coarse_payloads = add_to_dict(coarse_result, coarse_payloads)
            fine_payloads = add_to_dict(fine_result, fine_payloads)

        coarse_payloads = {k: tf.concat(v, 0) for k, v in coarse_payloads.items()}
        fine_payloads = {k: tf.concat(v, 0) for k, v in fine_payloads.items()}

        return coarse_payloads, fine_payloads

    @tf.function
    def train_step(
        self,
        ray_origins: tf.Tensor,
        ray_directions: tf.Tensor,
        camera_pose: tf.Tensor,
        near_bound: float,
        far_bound: float,
        illumination_idx: tf.Tensor,
        ev100: tf.Tensor,
        illumination_factor: tf.Tensor,
        is_wb_ref_image: tf.Tensor,
        wb_input_value: tf.Tensor,
        optimizer: tf.keras.optimizers.Optimizer,
        target: tf.Tensor,
        target_mask: tf.Tensor,
        lambda_advanced_loss: tf.Tensor,
        lambda_slow_fade_loss: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """Perform a single training step.

        Args:
            ray_origins (tf.Tensor(float32), [batch, 3]): the ray origin.
            ray_directions (tf.Tensor(float32), [batch, 3]): the ray direction.
            camera_pose (tf.Tensor(float32), [batch, 3, 3]): the camera matrix.
            near_bound (tf.Tensor(float32), [1]): the near clipping point.
            far_bound (tf.Tensor(float32), [1]): the far clipping point.
            illumination_idx (tf.Tensor(int32), [1]): the illumination index.
            ev100 (tf.Tensor(float32), [1]): the ev100 value of the image.
            is_wb_ref_image (tf.Tensor(bool) [1]): whether the current image is
                a reference whitebalance image.
            wb_input_value (tf.Tensor(float32) [1, 3]): if `is_wb_ref_image` then
                this is defines the whitebalance value.
            optimizer (tf.keras.optimizers.Optimizer): the optimizer to use in the
                train step.
            target (tf.Tensor(float32), [batch, 3]): the rgb target from the image.
            target_mask (tf.Tensor(float32), [batch, 1]): the segmentation mask
                target from the image.
            lambda_advanced_loss (tf.Tensor(float32), [1]): current advanced loss
                interpolation value.

        Returns:
            coarse_payload (Dict[str, tf.Tensor]): dict with the payload for the
                coarse network.
            fine_payload (Dict[str, tf.Tensor]): dict with the payload for the fine
                network.
            loss (tf.Tensor(float32), [1]): the joint loss.
            coarse_losses (Dict[str, tf.Tensor]): a dict of loss names with the
                evaluated losses. "loss" stores the final loss of the layer.
            fine_losses (Dict[str, tf.Tensor]): a dict of loss names with the evaluated
                losses. "loss" stores the final loss of the layer.
        """

        with tf.GradientTape() as tape:
            wb_loss = float(0)
            if is_wb_ref_image[0]:
                illumination_context = self.illumination_embedding_store(
                    illumination_idx
                    if not self.single_env
                    else tf.convert_to_tensor([0])
                )

                wb_scene = math_utils.saturate(
                    self.fine_model.get_white_balance_under_illumination(
                        illumination_context, ray_origins,
                    )
                    * illumination_factor
                    * math_utils.ev100_to_exp(ev100)
                )
                wb_loss = self.mse(wb_input_value, wb_scene)

            coarse_result, fine_result = self.call(
                ray_origins,
                ray_directions,
                camera_pose,
                near_bound,
                far_bound,
                illumination_idx,
                ev100,
                illumination_factor,
                training=True,
            )

            coarse_losses = self.coarse_model.calculate_losses(
                coarse_result, target, target_mask, lambda_advanced_loss
            )

            view_vector = math_utils.normalize(-1 * ray_directions)

            fine_losses = self.fine_model.calculate_losses(
                fine_result,
                target,
                target_mask,
                view_vector,
                lambda_advanced_loss,
                lambda_slow_fade_loss,
            )

            loss = coarse_losses["loss"] + fine_losses["loss"] + wb_loss

        grad_vars = (
            self.coarse_model.trainable_variables
            + self.fine_model.trainable_variables
            + self.illumination_embedding_store.trainable_variables
        )

        gradients = tape.gradient(loss, grad_vars)

        gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        optimizer.apply_gradients(zip(gradients, grad_vars))

        return loss, wb_loss, coarse_losses, fine_losses

    @tf.function
    def illumination_single_step(
        self,
        camera_pose,
        ray_directions,
        diffuse,
        specular,
        roughness,
        normal,
        alpha,
        illumination_idx,
        target,
        ev100,
        illumination_factor,
        mse,
        optimizer,
    ):
        with tf.name_scope("IlluminationSingleStep"):
            with tf.name_scope("Prepare"):
                is_background = alpha < 0.3
                select_on_background = lambda x, y: tf.where(
                    math_utils.repeat(is_background, tf.shape(x)[-1], -1), x, y,
                )

            with tf.name_scope("Directions"):
                view_directions = -1 * ray_directions
                org_viewdirections = view_directions
                (
                    view_directions,
                    reflection_direction,
                ) = self.fine_model.renderer.calculate_reflection_direction(
                    view_directions,
                    normal,
                    camera_pose=camera_pose[0] if self.rotating_object else None,
                )

                view_directions = select_on_background(
                    org_viewdirections, view_directions
                )

                reflection_direction = select_on_background(
                    math_utils.normalize(ray_directions), reflection_direction,
                )
                specular_roughness = select_on_background(
                    tf.zeros_like(roughness), roughness,
                )

            with tf.name_scope("Execute"):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(self.illumination_embedding_store.trainable_variables)

                    # Get current embedding
                    illumination_context = self.illumination_embedding_store(
                        illumination_idx
                        if not self.single_env
                        else tf.convert_to_tensor([0])
                    )

                    with tf.name_scope("Illumination"):
                        # Illumination net expects a B, S, C shape.
                        # Add a fake one and remove b dim afterward
                        diffuse_irradiance = self.fine_model.illumination_net.call_multi_samples(
                            reflection_direction[None, ...],
                            tf.ones_like(  # Just sample with maximum roughness
                                roughness[None, ...]
                            ),
                            illumination_context,
                        )[
                            0
                        ]

                        # Illumination net expects a B, S, C shape.
                        # Add a fake one and remove b dim afterward
                        specular_irradiance = self.fine_model.illumination_net.call_multi_samples(
                            reflection_direction[None, ...],
                            specular_roughness[None, ...],
                            illumination_context,
                        )[
                            0
                        ]

                    with tf.name_scope("Render"):
                        render = (
                            self.fine_model.renderer(
                                view_directions,
                                normal,
                                diffuse_irradiance,
                                specular_irradiance,
                                diffuse,
                                specular,
                                roughness,
                            )
                            * illumination_factor
                        )

                    with tf.name_scope("RenderPostProcess"):
                        # Replace background with illumination evaluation
                        render = select_on_background(specular_irradiance, render)

                        # Auto exposure + srgb to model camera setup
                        render = math_utils.white_background_compose(
                            self.fine_model.camera_post_processing(render, ev100), alpha
                        )

                    with tf.name_scope("Loss"):
                        illum_loss = mse(
                            math_utils.white_background_compose(
                                tf.reshape(target, (-1, 3)), alpha
                            ),
                            render,
                        )
                        tf.debugging.check_numerics(illum_loss, "loss illum")

                grad_vars = self.illumination_embedding_store.trainable_variables
                gradients = tape.gradient(illum_loss, grad_vars)

                optimizer.apply_gradients(zip(gradients, grad_vars))

            return illum_loss

    def illumination_steps(
        self,
        ray_origins: tf.Tensor,
        ray_directions: tf.Tensor,
        camera_pose: tf.Tensor,
        near_bound: float,
        far_bound: float,
        illumination_idx: tf.Tensor,
        ev100: tf.Tensor,
        illumination_factor: tf.Tensor,
        optimizer: tf.keras.optimizers.Optimizer,
        target: tf.Tensor,
        steps: int,
        chunk_size: int = 1024,
        strategy=tf.distribute.get_strategy(),
    ) -> tf.Tensor:
        """Perform a illumination optimization step. This only performs the illumination
        with a fixed network.

        Args:
            ray_origins (tf.Tensor(float32), [batch, 3]): the ray origin.
            ray_directions (tf.Tensor(float32), [batch, 3]): the ray direction.
            camera_pose (tf.Tensor(float32), [batch, 3, 3]): the camera matrix.
            near_bound (tf.Tensor(float32), [1]): the near clipping point.
            far_bound (tf.Tensor(float32), [1]): the far clipping point.
            illumination_idx (tf.Tensor(int32), [1]): the illumination index.
            ev100 (tf.Tensor(float32), [1]): the ev100 value of the image.
            optimizer (tf.keras.optimizers.Optimizer): the optimizer to use in the
                train step.
            target (tf.Tensor(float32), [batch, 3]): the rgb target from the image.
            steps (int): the number of optimization steps to perform.
            chunk_size (int): If specified runs the sampling in
                batches. Runs everything jointly if 0.

        Returns:
            tf.Tensor(float32), [1]: the loss after the optimization
        """

        mse = multi_gpu_wrapper(tf.keras.losses.MeanSquaredError, target.shape[0])

        _, fine_result = self.distributed_call(
            strategy,
            chunk_size,
            ray_origins,
            ray_directions,
            camera_pose,
            near_bound,
            far_bound,
            illumination_idx,
            ev100,
            illumination_factor,
            False,
        )

        data = [
            ray_directions,
            target,
            fine_result["diffuse"],
            fine_result["specular"],
            fine_result["roughness"],
            fine_result["normal"],
            fine_result["acc_alpha"][..., None],
        ]

        dp_df = tf.data.Dataset.from_tensor_slices((*data,)).batch(
            chunk_size * get_num_gpus()
        )
        dp_dist_df = strategy.experimental_distribute_dataset(dp_df)

        for i in tf.range(steps):
            total_loss = 0
            for dp in dp_dist_df:
                ray_d, trgt, diff, spec, rgh, nrm, alp = dp
                illum_loss_per_replica = strategy.run(
                    self.illumination_single_step,
                    (
                        camera_pose[:1],
                        ray_d,
                        diff,
                        spec,
                        rgh,
                        nrm,
                        alp,
                        illumination_idx[:1],
                        trgt,
                        ev100,
                        illumination_factor,
                        mse,
                        optimizer,
                    ),
                )
                illum_loss = strategy.reduce(
                    tf.distribute.ReduceOp.SUM, illum_loss_per_replica, axis=None
                )
                total_loss = total_loss + illum_loss

        return total_loss

    def calculate_illumination_factor(
        self, camera_position, ev100_target, illumination_context_overwrite=None
    ):
        if illumination_context_overwrite is None:
            illumination_context = self.illumination_embedding_store.latent_mean[
                None, :
            ]
        else:
            illumination_context = illumination_context_overwrite
        lum_volume = tf.reduce_mean(
            self.fine_model.get_white_balance_under_illumination(
                illumination_context, camera_position
            )
        )

        target = 0.8 / tf.maximum(math_utils.ev100_to_exp(ev100_target), 1e-5)

        factor = target / lum_volume
        return factor

    @classmethod
    def add_args(cls, parser):
        """Add the base nerf arguments to the parser with addition
        to the specific Neural-PIL ones.

        Args:
            parser (ArgumentParser): the current ArgumentParser.

        Returns:
            ArgumentParser: the modified ArgumentParser for call chaining
        """
        add_base_args(parser)
        parser.add_argument(
            "--coarse_samples",
            type=int,
            default=64,
            help="number of coarse samples per ray in a fixed grid",
        )
        parser.add_argument(
            "--fine_samples",
            type=int,
            default=128,
            help="number of additional samples per ray based on the coarse samples",
        )
        parser.add_argument(
            "--fourier_frequency",
            type=int,
            default=10,
            help="log2 of max freq for positional encoding",
        )
        parser.add_argument(
            "--net_width", type=int, default=256, help="channels per layer"
        )
        parser.add_argument(
            "--net_depth", type=int, default=8, help="layers in network"
        )

        # Illumination configs
        parser.add_argument(
            "--rotating_object",
            action="store_true",
            help=(
                "The object is rotating instead of the camera. The illumination then "
                "needs to stay static"
            ),
        )
        parser.add_argument(
            "--single_env",
            action="store_true",
            help="All input images are captured under a single environment",
        )

        # Render configs

        parser.add_argument(
            "--brdf_preintegration_path",
            default="data/neural_pil/BRDFLut.hdr",
            help="Path to the preintegrated BRDF LUT.",
        )

        # Coarse configs
        parser.add_argument(
            "-lindisp",
            "--linear_disparity_sampling",
            action="store_true",
            help="Coarse sampling linearly in disparity rather than depth",
        )

        # Fine configs

        parser.add_argument(
            "--brdf_network_path",
            default="data/neural_pil/brdf-network",
            help="Path to the BRDF decoder config and weights",
        )
        parser.add_argument(
            "--illumination_network_path",
            default="data/neural_pil/illumination-network",
            help="Path to the illumination network config and weights",
        )

        parser.add_argument(
            "--direct_rgb",
            action="store_true",
            help=(
                "Also performs a direct RGB color prediction. This is useful in the "
                "beginning of the training."
            ),
        )

        parser.add_argument(
            "--advanced_loss_done",
            type=int,
            default=60000,
            help=(
                "Exponentially decays losses. After this many steps the loss is reduced"
                "by 3 magnitudes"
            ),
        )

        parser.add_argument("--ablate_brdf_smae", action="store_true")

        return parser
