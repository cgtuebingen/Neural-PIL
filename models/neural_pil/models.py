import os
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

import nn_utils.math_utils as math_utils
from losses import multi_gpu_wrapper
from losses.nerd import cosine_weighted_mse, segmentation_mask_loss
from models.brdf_ae_net.models import BrdfDecoder
from models.illumination_integration_net import IlluminationNetwork
from nn_utils.nerf_layers import (
    FourierEmbedding,
    add_gaussian_noise,
    setup_fixed_grid_sampling,
    setup_hierachical_sampling,
    split_sigma_and_payload,
    volumetric_rendering,
)
from nn_utils.preintegrated_rendering import PreintegratedRenderer
from train_brdf_ae import parser as brdf_parser
from train_illumination_net import parser as illumination_parser
from utils.training_setup_utils import get_num_gpus


class CoarseModel(tf.keras.Model):
    def __init__(self, illumination_latent_dim, args, **kwargs):
        super(CoarseModel, self).__init__(**kwargs)

        self.num_samples = args.coarse_samples
        self.linear_disparity_sampling = args.linear_disparity_sampling
        self.raw_noise_std = args.raw_noise_std

        self.illumination_latent_dim = illumination_latent_dim

        # Start with fourier embedding
        self.pos_embedder = FourierEmbedding(args.fourier_frequency)
        main_net = [
            tf.keras.layers.InputLayer(
                (self.pos_embedder.get_output_dimensionality(),)
            ),
        ]
        # Then add the main layers
        for _ in range(args.net_depth // 2):
            main_net.append(tf.keras.layers.Dense(args.net_width, activation="relu",))
        # Build network stack
        self.main_net_first = tf.keras.Sequential(main_net)

        main_net = [
            tf.keras.layers.InputLayer(
                (args.net_width + self.pos_embedder.get_output_dimensionality(),)
            ),
        ]
        for _ in range(args.net_depth // 2):
            main_net.append(tf.keras.layers.Dense(args.net_width, activation="relu",))
        self.main_net_second = tf.keras.Sequential(main_net)

        # Sigma is a own output not conditioned on the illumination
        self.sigma_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((args.net_width,)),
                tf.keras.layers.Dense(1, activation="linear"),
            ]
        )
        print("Coarse sigma\n", self.sigma_net.summary())

        # Build a small conditional net which gets the embedding from the main net
        # plus the illumination
        self.conditional_embedding = FourierEmbedding(4)
        self.illumination_reduce = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((self.illumination_latent_dim,)),
                tf.keras.layers.Dense(32, activation="linear"),
            ]
        )
        self.bottle_neck_layer = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((args.net_width,)),
                tf.keras.layers.Dense(args.net_width, activation="relu",),
            ]
        )
        self.conditional_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    (
                        args.net_width
                        + 32
                        + self.conditional_embedding.get_output_dimensionality(),
                    )
                ),
                tf.keras.layers.Dense(args.net_width, activation="relu",),
                tf.keras.layers.Dense(3, activation="linear"),
            ]
        )
        print("Coarse conditional\n", self.conditional_net.summary())

        self.num_gpu = max(1, get_num_gpus())
        self.global_batch_size = args.batch_size * self.num_gpu
        self.mse = multi_gpu_wrapper(
            tf.keras.losses.MeanSquaredError, self.global_batch_size,
        )
        self.alpha_loss = segmentation_mask_loss(self.global_batch_size,)

    def payload_to_parmeters(
        self, raymarched_payload: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        ret = {"rgb": tf.sigmoid(raymarched_payload)}  # Just RGB here

        for k in ret:
            tf.debugging.check_numerics(ret[k], "output {}".format(k))
        return ret

    @tf.function
    def call(
        self,
        pts: tf.Tensor,
        illumination_context: tf.Tensor,
        ray_directions: tf.Tensor,
        randomized: bool = False,
    ) -> tf.Tensor:
        """Evaluates the network for all points and condition it on the illumination

        Args:
            pts (tf.Tensor(float32), [..., 3]): the points where to evaluate the
                network.
            illumination_context (tf.Tensor(float32), [1, illumination_latent_dim]): the
                embedding for the images.
            ray_directions (tf.Tensor(float32), [..., 3]): the ray directions for
                each sample.
            randomized (bool): use randomized sigma noise. Defaults to False.

        Returns:
            sigma_payload (tf.Tensor(float32), [..., 1 + 3]): the sigma and the
                rgb.
        """
        pts_embed = self.pos_embedder(tf.reshape(pts, (-1, pts.shape[-1])))
        main_embd = self.main_net_first(
            pts_embed
        )  # Run points through main net. The embedding is flat B*S, C
        main_embd = self.main_net_second(tf.concat([main_embd, pts_embed], -1))

        # Extract sigma
        sigma = self.sigma_net(main_embd)
        sigma = add_gaussian_noise(sigma, self.raw_noise_std, randomized)

        # Prepare the illrandomumination context to fit the shape
        # Do not optimize the context from the coarse network
        illumination_context = tf.stop_gradient(
            math_utils.normalize(illumination_context)
        )
        # Ensure value range
        illumination_context = self.illumination_reduce(
            illumination_context
        ) * tf.ones_like(main_embd[:, :1])

        # View dependent
        ray_embd = self.conditional_embedding(
            tf.reshape(
                tf.reshape(
                    ray_directions,
                    (
                        tf.shape(ray_directions)[0],
                        *[1 for _ in pts.shape[1:-1]],
                        ray_directions.shape[-1],
                    ),
                )
                * tf.ones_like(pts),
                (-1, ray_directions.shape[-1]),
            )
        )

        # Concat main embedding and the context
        main_embd = self.bottle_neck_layer(main_embd)
        conditional_input = tf.concat([main_embd, illumination_context, ray_embd], -1)
        # Predict the conditional RGB
        rgb = self.conditional_net(conditional_input)

        # Build the final output
        sigma_payload_flat = tf.concat([sigma, rgb], -1)
        new_shape = tf.concat([tf.shape(pts)[:-1], sigma_payload_flat.shape[-1:]], 0)
        return tf.reshape(sigma_payload_flat, new_shape)

    @tf.function
    def render_rays(
        self,
        ray_origins: tf.Tensor,
        ray_directions: tf.Tensor,
        near_bound: float,
        far_bound: float,
        illumination_context: tf.Tensor,
        randomized: bool = False,
        overwrite_num_samples: Optional[int] = None,
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor, tf.Tensor]:
        """Render the rays

        Args:
            ray_origins (tf.Tensor(float32), [batch, 3]): the ray origin.
            ray_directions (tf.Tensor(float32), [batch, 3]): the ray direction.
            near_bound (float): the near clipping point.
            far_bound (float): the far clipping point.
            illumination_context (tf.Tensor(float32), [1, illumination_latent_dim]): the
                embedding for the images.
            randomized (bool, optional): Activates noise and pertub ray features.
                Defaults to False.

        Returns:
            payload (Dict[str, tf.Tensor(float32) [batch, num_channels]]): the
                raymarched payload dictionary.
            z_samples (tf.Tensor(float32), [batch, samples]): the distances to sample
                along the ray.
            weights (tf.Tensor(float32) [batch, num_samples]): the weights along the
                ray. That is the accumulated product of the individual alphas.
        """
        points, z_samples = setup_fixed_grid_sampling(
            ray_origins,
            ray_directions,
            near_bound,
            far_bound,
            self.num_samples
            if overwrite_num_samples is None
            else overwrite_num_samples,
            randomized=randomized,
            linear_disparity=self.linear_disparity_sampling,
        )

        raw = self.call(
            points, illumination_context, ray_directions, randomized=randomized,
        )

        sigma, payload_raw = split_sigma_and_payload(raw)

        payload, weights = volumetric_rendering(
            sigma,
            payload_raw,
            z_samples,
            ray_directions,
            self.payload_to_parmeters,
            ["rgb"],
        )

        return payload, z_samples, weights

    @tf.function
    def calculate_losses(
        self,
        payload: Dict[str, tf.Tensor],
        target: tf.Tensor,
        target_mask: tf.Tensor,
        lambda_advanced_loss: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Calculates the losses

        Args:
            payload (Dict[str, tf.Tensor(float32) [batch, num_channels]]): the
                raymarched payload dictionary.
            target (tf.Tensor(float32), [batch, 3]): the RGB target of the
                respective ray
            target_mask (tf.Tensor(float32), [batch, 1]): the segmentation mask
                target for the respective ray.

        Returns:
            Dict[str, tf.Tensor]: a dict of loss names with the evaluated losses.
                "loss" stores the final loss of the layer.
        """
        inverse_advanced = 1 - lambda_advanced_loss

        target_masked = tf.stop_gradient(
            math_utils.white_background_compose(target, target_mask)
        )

        alpha_loss = self.alpha_loss(
            payload["individual_alphas"],
            payload["acc_alpha"][..., None],
            target_mask,
            0,
        )

        image_loss = self.mse(target_masked, payload["rgb"])

        final_loss = image_loss + alpha_loss * 0.4 * inverse_advanced

        losses = {
            "loss": final_loss,
            "image_loss": image_loss,
            "alpha_loss": alpha_loss,
        }

        for k in losses:
            tf.debugging.check_numerics(losses[k], "loss {}".format(k))

        return losses


class FineModel(tf.keras.Model):
    def __init__(self, args, **kwargs):
        super(FineModel, self).__init__(**kwargs)

        self.num_samples = args.fine_samples
        self.coarse_samples = args.coarse_samples
        self.raw_noise_std = args.raw_noise_std

        self.direct_rgb = args.direct_rgb

        self.rotating_object = args.rotating_object

        self.ablate_brdf_smae = args.ablate_brdf_smae

        # Start with fourier embedding
        self.pos_embedder = FourierEmbedding(args.fourier_frequency)
        main_net = [
            tf.keras.layers.InputLayer(
                (self.pos_embedder.get_output_dimensionality(),)
            ),
        ]
        # Then add the main layers
        for _ in range(args.net_depth // 2):
            main_net.append(tf.keras.layers.Dense(args.net_width, activation="relu",))
        # Build network stack
        self.main_net_first = tf.keras.Sequential(main_net)

        main_net = [
            tf.keras.layers.InputLayer(
                (args.net_width + self.pos_embedder.get_output_dimensionality(),)
            ),
        ]
        for _ in range(args.net_depth // 2):
            main_net.append(tf.keras.layers.Dense(args.net_width, activation="relu",))
        self.main_net_second = tf.keras.Sequential(main_net)

        # Build the BRDF decoder
        if not self.ablate_brdf_smae:
            brdf_decoder_parser = brdf_parser()
            brdf_args = brdf_decoder_parser.parse_args(
                args="--config %s" % os.path.join(args.brdf_network_path, "args.txt")
            )

            self.brdf_decoder = BrdfDecoder(brdf_args, trainable=False)
            self.brdf_decoder(tf.zeros((1, brdf_args.latent_dim)))
            path = os.path.join(args.brdf_network_path, "decoder.npy")
            self.brdf_decoder.set_weights(np.load(path, allow_pickle=True))

            self.brdf_mean_std = np.load(
                os.path.join(args.brdf_network_path, "brdf_latent_mean_std.npy")
            )

        # Build the Illumination network
        illum_parser = illumination_parser()
        illum_args = illum_parser.parse_args(
            args="--config %s"
            % os.path.join(args.illumination_network_path, "args.txt")
        )

        illumination_net = IlluminationNetwork(illum_args, trainable=False)
        path = os.path.join(args.illumination_network_path, "network.npy")

        self.illumination_net = illumination_net.illumination_network
        self.illumination_net.set_weights(np.load(path, allow_pickle=True))

        # Extract interesting networks
        self.illum_main_net = self.illumination_net.main_network
        self.illum_conditional_mapping_net = self.illumination_net.conditional_network
        self.illum_mapping_net = self.illumination_net.mapping_network

        # Add a final layer for the main net which predicts sigma plus eventual
        # direct_rgb or mlp_normals
        self.main_final_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(args.net_width),
                tf.keras.layers.Dense(
                    # Decide how many features we need (at least 1 - sigma)
                    1
                    + (3 if args.direct_rgb else 0)
                    + (
                        self.brdf_decoder.latent_dim if not self.ablate_brdf_smae else 7
                    ),  # The BRDF embedding
                    activation="linear",
                ),
            ]
        )
        print("Fine final\n", self.main_final_net.summary())

        # Add the renderer
        self.renderer = PreintegratedRenderer(args.brdf_preintegration_path)

        # Add losses
        self.num_gpu = max(1, get_num_gpus())
        self.global_batch_size = args.batch_size * self.num_gpu

        self.mse = multi_gpu_wrapper(
            tf.keras.losses.MeanSquaredError, self.global_batch_size,
        )
        self.cosine_mse = cosine_weighted_mse(self.global_batch_size)
        self.alpha_loss = segmentation_mask_loss(self.global_batch_size,)

    def payload_to_parmeters(
        self, raymarched_payload: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        ret = {}
        start = 0  # Index where the extraction starts
        if self.direct_rgb:
            ret["direct_rgb"] = tf.sigmoid(raymarched_payload[..., start : start + 3])
            start += 3

        # Ensure the value range is -1 to 1 if mlp normals are used

        ret["normal"] = math_utils.normalize(raymarched_payload[..., start : start + 3])
        start += 3

        # BRDF parameters
        if not self.ablate_brdf_smae:
            brdf_activation = tf.identity
        else:
            brdf_activation = tf.nn.sigmoid

        ret["diffuse"] = brdf_activation(raymarched_payload[..., start : start + 3])
        start += 3
        ret["specular"] = brdf_activation(raymarched_payload[..., start : start + 3])
        start += 3
        ret["roughness"] = brdf_activation(raymarched_payload[..., start : start + 1])
        start += 1

        # Energy conservation and plausibility
        ret["diffuse"] = tf.clip_by_value(ret["diffuse"], 0 / 255, 240 / 255)
        lin_diffuse = math_utils.srgb_to_linear(ret["diffuse"])
        lin_specular = math_utils.srgb_to_linear(ret["specular"])
        max_sum = tf.maximum(
            tf.reduce_max(lin_specular + lin_diffuse, -1, keepdims=True), 1.0
        )
        lin_diffuse = tf.maximum(lin_diffuse + (1 - max_sum), 0)
        ret["diffuse"] = math_utils.linear_to_srgb(lin_diffuse)

        for k in ret:
            tf.debugging.check_numerics(ret[k], "output {}".format(k))

        return ret

    @tf.function
    def call(self, pts, randomized=False) -> tf.Tensor:
        """Evaluates the network for all points

        Args:
            pts (tf.Tensor(float32), [..., 3]): the points where to evaluate
                the network.
            view_direction (tf.Tensor(float32), [..., 3]): View direction pointing
                from the surface to the camera.
            randomized (bool): use randomized sigma noise. Defaults to False.

        Returns:
            sigma_payload (tf.Tensor(float32), [..., samples 1 + payload_channels]):
                the sigma and the payload.
        """
        # Tape to calculate the normal gradient
        with tf.GradientTape(watch_accessed_variables=False) as normalTape:
            normalTape.watch(pts)  # Watch pts as it is not a variable

            pts_flat = tf.reshape(pts, (-1, pts.shape[-1]))
            pts_embed = self.pos_embedder(pts_flat)

            # Call the main network
            main_embd = self.main_net_first(pts_embed)
            main_embd = self.main_net_second(tf.concat([main_embd, pts_embed], -1))

            # Split sigma and payload
            sigma_payload = self.main_final_net(main_embd)
            sigma = sigma_payload[..., :1]

            payload = sigma_payload[..., 1:]

        # Build payload list

        # Start with direct rgb if present
        start = 3 if self.direct_rgb else 0
        full_payload_list = [payload[..., :start]]

        # Normals are not directly predicted
        # Normals are derived from the gradient of sigma wrt. to the input points
        normal = math_utils.normalize(-1 * normalTape.gradient(sigma, pts_flat))
        full_payload_list.append(normal)

        # Evaluate the BRDF
        if not self.ablate_brdf_smae:
            brdf_embedding = payload[
                ..., start : start + self.brdf_decoder.latent_dim,
            ]
            start = start + self.brdf_decoder.latent_dim

            # Scale with training set mean/var
            brdf_embedding = (
                brdf_embedding * (self.brdf_mean_std[1:2] * 3) + self.brdf_mean_std[0:1]
            )
            brdf = self.brdf_decoder(brdf_embedding)
            full_payload_list.append(brdf)
        else:
            full_payload_list.append(
                payload[..., start : start + 7,]
            )
            start = start + 7

        # Add noise
        sigma = add_gaussian_noise(sigma, self.raw_noise_std, randomized)

        # Build the output sigma and payload
        sigma_payload_flat = tf.concat([sigma] + full_payload_list, -1)
        new_shape = tf.concat([tf.shape(pts)[:-1], sigma_payload_flat.shape[-1:]], 0)
        return tf.reshape(sigma_payload_flat, new_shape)

    def get_white_balance_under_illumination(
        self, illumination_context, camera_position, ev100: Optional[tf.Tensor] = None
    ):
        normal_dir = math_utils.normalize(camera_position[:1])

        # Illumination net expects a B, S, C shape.
        # Add a fake one and remove s dim afterward
        diffuse_irradiance = self.illumination_net.call_multi_samples(
            tf.expand_dims(normal_dir, 1),
            tf.expand_dims(
                tf.ones_like(normal_dir[:, :1]), 1
            ),  # Just sample with maximum roughness
            illumination_context,
        )[:, 0]

        # Illumination net expects a B, S, C shape.
        # Add a fake one and remove s dim afterward
        specular_irradiance = self.illumination_net.call_multi_samples(
            tf.expand_dims(normal_dir, 1),
            tf.expand_dims(tf.ones_like(normal_dir[:, :1]), 1),
            illumination_context,
        )[:, 0]

        rgb = self.renderer(
            normal_dir,
            normal_dir,
            diffuse_irradiance,
            specular_irradiance,
            tf.constant([[0.8, 0.8, 0.8]], dtype=tf.float32),
            tf.constant([[0.04, 0.04, 0.04]], dtype=tf.float32),
            tf.constant([[1.0]], dtype=tf.float32),
        )

        if ev100 is not None:
            exp_val = tf.stop_gradient(math_utils.ev100_to_exp(ev100))
            rgb = rgb * exp_val

        return rgb

    @tf.function
    def render_rays(
        self,
        ray_origins: tf.Tensor,
        ray_directions: tf.Tensor,
        previous_z_samples: tf.Tensor,
        previous_weights: tf.Tensor,
        camera_pose: tf.Tensor,
        illumination_context: tf.Tensor,
        ev100: tf.Tensor,
        illumination_factor: tf.Tensor,
        randomized: bool = False,
        overwrite_num_samples: Optional[int] = None,
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor, tf.Tensor]:
        """Render the rays

        Args:
            ray_origins (tf.Tensor(float32), [batch, 3]): the ray origin.
            ray_directions (tf.Tensor(float32), [batch, 3]): the ray direction.
            previous_z_samples (tf.Tensor(float32), [batch, samples]): the previous
                distances to sample along the ray.
            previous_weights (tf.Tensor(float32) [batch, num_samples]): the previous
                weights along the ray. That is the accumulated product of the
                individual alphas.
            camera_pose (tf.Tensor(float32), [batch, 3, 4]): the camera matrix.
            illumination_context (tf.Tensor(float32), [batch, illumination_latent_dim]):
                the embedding for the images.
            ev100 (tf.Tensor(float32), [1]): the ev100 value of the image.
            randomized (bool, optional): Activates noise and pertub ray features.
                Defaults to False.

        Returns:
            payload (Dict[str, tf.Tensor(float32) [batch, num_channels]]): the
                raymarched payload dictionary.
            z_samples (tf.Tensor(float32), [batch, samples]): the distances to sample
                along the ray.
            weights (tf.Tensor(float32) [batch, num_samples]): the weights along the
                ray. That is the accumulated product of the individual alphas.
        """
        points, z_samples = setup_hierachical_sampling(
            ray_origins,
            ray_directions,
            previous_z_samples,
            previous_weights,
            self.num_samples
            if overwrite_num_samples is None
            else overwrite_num_samples,
            randomized=randomized,
        )

        # View direction is inverse ray direction. Which points from surface to camera
        view_direction = math_utils.normalize(-1 * ray_directions)

        raw = self.call(points, randomized=randomized)

        sigma, payload_raw = split_sigma_and_payload(raw)

        payload, weights = volumetric_rendering(
            sigma,
            payload_raw,
            z_samples,
            ray_directions,
            self.payload_to_parmeters,
            ["diffuse", "specular", "roughness"]
            + (
                ["direct_rgb"] if self.direct_rgb else []
            ),  # Check if direct rgb is requested
        )

        # Ensure the raymarched normal is actually normalized
        payload["normal"] = math_utils.white_background_compose(
            math_utils.normalize(payload["normal"]), payload["acc_alpha"][:, None],
        )

        # First get the reflection direction
        # Add a fake sample dimension
        (
            view_direction,
            reflection_direction,
        ) = self.renderer.calculate_reflection_direction(
            view_direction,
            payload["normal"],
            camera_pose=(camera_pose[0] if len(camera_pose.shape) == 3 else camera_pose)
            if self.rotating_object and camera_pose is not None
            else None,
        )

        # Illumination net expects a B, S, C shape.
        diffuse_irradiance = self.illumination_net.call_multi_samples(
            tf.expand_dims(reflection_direction, 0),
            tf.expand_dims(
                tf.ones_like(
                    payload["roughness"]
                ),  # Just sample with maximum roughness
                0,
            ),
            illumination_context,
        )[0]

        # Illumination net expects a B, S, C shape.
        # Add a fake one and remove s dim afterward
        specular_irradiance = self.illumination_net.call_multi_samples(
            tf.expand_dims(reflection_direction, 0),
            tf.expand_dims(payload["roughness"], 0),
            illumination_context,
        )[0]

        # Everything now should be B*S. Make sure that shapes
        # are okay
        rgb = (
            self.renderer(
                view_direction,
                payload["normal"],
                diffuse_irradiance,
                specular_irradiance,
                payload["diffuse"],
                payload["specular"],
                payload["roughness"],
            )
            * illumination_factor
        )

        payload["hdr_rgb"] = rgb

        payload["rgb"] = math_utils.white_background_compose(
            self.camera_post_processing(rgb, ev100), payload["acc_alpha"][..., None],
        )

        return (
            payload,
            z_samples,
            weights,
        )

    @tf.function
    def camera_post_processing(self, hdr_rgb: tf.Tensor, ev100: tf.Tensor) -> tf.Tensor:
        """Applies the camera auto-exposure post-processing

        Args:
            hdr_rgb (tf.Tensor(float32), [..., 3]): the HDR input fromt the
                rendering step.
            ev100 ([type]): [description]

        Returns:
            tf.Tensor: [description]
        """
        exp_val = tf.stop_gradient(math_utils.ev100_to_exp(ev100))
        ldr_rgb = math_utils.linear_to_srgb(math_utils.saturate(hdr_rgb * exp_val))

        return ldr_rgb

    @tf.function
    def calculate_losses(
        self,
        payload: Dict[str, tf.Tensor],
        target: tf.Tensor,
        target_mask: tf.Tensor,
        view_vector: tf.Tensor,
        lambda_advanced_loss: tf.Tensor,
        lambda_slow_fade_loss: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Calculates the losses

        Args:
            payload (Dict[str, tf.Tensor(float32) [batch, num_channels]]): the
                raymarched payload dictionary.
            target (tf.Tensor(float32), [batch, 3]): the RGB target of the
                respective ray
            target_mask (tf.Tensor(float32), [batch, 1]): the segmentation mask
                target for the respective ray.
            lambda_advanced_loss (tf.Tensor(float32), [1]): current advanced loss
                interpolation value.

        Returns:
            Dict[str, tf.Tensor]: a dict of loss names with the evaluated losses.
                "loss" stores the final loss of the layer.
        """
        lambda_inverse_advanced_loss = (
            1 - lambda_advanced_loss
        )  # Starts with 0 goes to 1
        lambda_inverse_slow_fade_loss = 1 - lambda_slow_fade_loss

        target_masked = tf.stop_gradient(
            math_utils.white_background_compose(target, target_mask)
        )

        alpha_loss = self.alpha_loss(
            payload["individual_alphas"],
            payload["acc_alpha"][..., None],
            target_mask,
            0,
        )

        # Calculate losses
        direct_img_loss = 0
        if self.direct_rgb:
            direct_img_loss = self.mse(target_masked, payload["direct_rgb"])

        image_mse_loss = self.mse(target_masked, payload["rgb"],)

        cos_theta = math_utils.dot(payload["normal"], view_vector)
        cos_theta = tf.stop_gradient(
            math_utils.white_background_compose(cos_theta, target_mask)
        )  # Ensure loss in background is always weighted fully
        image_cosine_loss = self.cosine_mse(target_masked, payload["rgb"], cos_theta)

        image_loss = (
            image_mse_loss  # + image_cosine_loss * lambda_inverse_slow_fade_loss
        )

        diffuse_initial_loss = self.mse(target_masked, payload["diffuse"])
        roughness_initial_loss = self.mse(
            tf.stop_gradient(
                math_utils.white_background_compose(
                    tf.ones_like(payload["roughness"]) * 0.4, target_mask
                )
            ),
            payload["roughness"],
        )

        final_loss = (
            image_loss * tf.maximum(lambda_inverse_slow_fade_loss, 0.1)
            + alpha_loss * lambda_inverse_advanced_loss
            + direct_img_loss * lambda_slow_fade_loss
            + 4 * diffuse_initial_loss * lambda_slow_fade_loss
            + 0.25 * roughness_initial_loss * lambda_slow_fade_loss
        )

        losses = {
            "loss": final_loss,
            "image_loss": image_loss,
            "alpha_loss": alpha_loss,
            "diffuse_initial_loss": diffuse_initial_loss,
            "roughness_initial_loss": roughness_initial_loss,
        }

        if self.direct_rgb:
            losses["direct_rgb_loss"] = direct_img_loss

        for k in losses:
            tf.debugging.check_numerics(losses[k], "loss {}".format(k))

        return losses
