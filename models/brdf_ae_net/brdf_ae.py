from typing import Optional

import tensorflow as tf

from losses.brdf_ae import (
    CyclicEmbeddingLoss,
    DistanceLoss,
    LsganDiscriminatorLoss,
    lsganGeneratorLoss,
    smoothnessLoss,
)
from models.brdf_ae_net.models import BrdfDecoder, BrdfDiscriminator, BrdfEncoder
from nn_utils.utils_layers import Interpolation
from utils.training_setup_utils import StateRestoration, StateRestorationItem


class BrdfInterpolatingAutoEncoder(tf.keras.Model):
    def __init__(
        self, args, **kwargs,
    ):
        super(BrdfInterpolatingAutoEncoder, self).__init__(**kwargs)

        self.no_roughness = args.no_roughness
        input_dimensions = 6 if args.no_roughness else 7

        self.encoder = BrdfEncoder(args, input_dimensions=input_dimensions)
        self.decoder = BrdfDecoder(args, output_dimensions=input_dimensions)
        self.discriminator = BrdfDiscriminator(args, input_dimensions=input_dimensions)

        states = [
            StateRestorationItem("encoder", self.encoder),
            StateRestorationItem("decoder", self.decoder),
            StateRestorationItem("discriminator", self.discriminator),
        ]
        self.state_restoration = StateRestoration(args, states)

        self.interpolator = Interpolation()

        self.lambda_generator_loss = args.lambda_generator_loss
        self.lambda_cyclic_loss = args.lambda_cyclic_loss
        self.lambda_smoothness_loss = args.lambda_smoothness_loss
        self.lambda_distance_loss = args.lambda_distance_loss

        self.optimizer = tf.keras.optimizers.Adam(args.learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(args.learning_rate)

        self.interpolation_samples = args.interpolation_samples

        self.mae = lambda true, pred: tf.reduce_mean(
            tf.math.abs(true - pred)
        ) + tf.reduce_mean(tf.math.square(true - pred))
        self.smoothness_loss = smoothnessLoss
        self.cycle_loss = CyclicEmbeddingLoss()
        self.generator_loss = lsganGeneratorLoss
        self.discriminator_loss = LsganDiscriminatorLoss()
        self.distance_loss = DistanceLoss()

    def save(self, step):
        # Save weights for step
        self.state_restoration.save(step)

    def restore(self, step: Optional[int] = None) -> int:
        # Restore weights from step or if None the latest one
        return self.state_restoration.restore(step)

    @tf.function
    def train_step(self, x):
        # First the generator
        if self.no_roughness:
            x = x[..., :6]

        with tf.GradientTape(persistent=True) as tape:
            z = self.encoder(x)
            x_recon = self.decoder(z)

            # Direct reconstruction
            reconstruction_loss = self.mae(x, x_recon)

            # Setup smoothness and cyclic loss
            interpolation_values = tf.linspace(0.0, 1.0, self.interpolation_samples)
            with tf.GradientTape() as interpolTape:
                interpolTape.watch([interpolation_values, z])

                zfh, zsh = tf.split(z, 2)

                z_interpol = self.interpolator.multiple_alphas(
                    zfh, zsh, interpolation_values
                )
                # Sample dimension in second place
                z_interpol = tf.transpose(z_interpol, [1, 0, 2])
                z_interpol = tf.reshape(
                    z_interpol, (-1, z.shape[-1])
                )  # Flatten to B*T batch dim
                interpolated_samples = self.decoder(z_interpol)

            # Smoothness loss
            interpolation_gradient = interpolTape.gradient(
                interpolated_samples, interpolation_values,
            )
            smoothness_loss = (
                self.smoothness_loss(interpolation_gradient) / zfh.shape[0]
            )

            # Cyclic loss

            cycle_z_interpolated = self.encoder(interpolated_samples)
            cycle_loss = self.cycle_loss(z_interpol, cycle_z_interpolated)

            # Setup GAN losses
            fake_logits = self.discriminator(interpolated_samples)
            real_logits = self.discriminator(x)

            # Generator loss
            generator_loss = self.generator_loss(fake_logits)

            # Discriminator loss
            discriminator_loss = self.discriminator_loss(real_logits, fake_logits)

            # Distance loss
            distance_loss = self.distance_loss(x, z)

            # Final weighted loss
            loss = (
                reconstruction_loss
                + self.lambda_generator_loss * generator_loss
                + self.lambda_cyclic_loss * cycle_loss
                + self.lambda_smoothness_loss * smoothness_loss
                + self.lambda_distance_loss * distance_loss
            )

        # Optimize the generator part (Encoder + decoder)
        grad_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, grad_vars)
        self.optimizer.apply_gradients(zip(gradients, grad_vars))

        # Now optimize the discriminator
        grad_vars = self.discriminator.trainable_variables
        gradients = tape.gradient(discriminator_loss, grad_vars)
        self.discriminator_optimizer.apply_gradients(zip(gradients, grad_vars))

        losses = {
            "reconstruction_loss": reconstruction_loss,
            "distance_loss": distance_loss,
            "discriminator_loss": discriminator_loss,
            "generator_loss": generator_loss,
            "cycle_loss": cycle_loss,
            "smoothness_loss": smoothness_loss,
            "loss": loss,
        }
        return (x_recon, interpolated_samples, z, losses)

    def test_step(self, x):
        if self.no_roughness:
            x = x[..., :6]

        z = self.encoder(x)
        x_recon = self.decoder(z)

        mse = tf.keras.losses.mean_squared_error(x, x_recon)

        return (x, x_recon, mse)
