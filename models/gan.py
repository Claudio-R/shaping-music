import os, sys
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import PIL
from models.multimodal_feature_extractor import MultimodalFeatureExtractor
from models.image_to_sound_encoder import ImageToSoundEncoder
from models.sound_to_image_encoder import SoundToImageEncoder
from models.generator import Generator
from models.discriminator import Discriminator

class GenerativeAdversarialNetwork():
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.mfe = MultimodalFeatureExtractor()
        self.i2sEncoder = ImageToSoundEncoder()
        self.i2sEncoder.load_weights('data/weights/image_to_sound_encoder.h5')

        self.s2iEncoder = SoundToImageEncoder()
        self.s2iEncoder.load_weights('data/weights/sound_to_image_encoder.h5')

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.checkpoint_dir = 'data/checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "gan_ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
            )
    
    # TODO: Convert this to a tf.function and fix bug in __load_image()
    def train(self, audio_urls, epochs, seed):
        for epoch in range(epochs):
            print("Epoch: {}".format(epoch))
            for audio_batch in audio_urls:
                self.__train_step(audio_batch)

            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
            
            print("Generating and saving images")
            img = self.generator(seed, training=False)
            img = (img[0, :, :, :] * 127.5 + 127.5).numpy().astype(np.uint8)
            PIL.Image.fromarray(img).save('data/generated_images/image_at_epoch_{:04d}.png'.format(epoch))
                
        self.generator.save_weights('data/weights/gan_generator.h5')
        self.discriminator.save_weights('data/weights/gan_discriminator.h5')
        
    def __train_step(self, audio_batch, noise_dim=100):
        noise = tf.random.normal([len(audio_batch), noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_tape.watch(self.generator.trainable_variables)
            disc_tape.watch(self.discriminator.trainable_variables)

            # generate an image
            generated_image = self.generator(noise, training=True)
            generated_image = tf.expand_dims(generated_image[0, :, :, :], 0)

            # extract image embeddings from generated image
            generated_image_embeds = self.mfe.predict_from_image(generated_image)
            generated_image_embeds = tf.expand_dims(generated_image_embeds, 0)

            # encode image embeddings to sound embeddings
            generated_audio_embeds = self.i2sEncoder(generated_image_embeds, training=False)

            # extract audio embeddings from original audio
            original_audio_embeds = self.mfe('sound', audio_batch)

            # feed audio features to discriminator
            real_output = self.discriminator(original_audio_embeds, training=True)
            fake_output = self.discriminator(generated_audio_embeds, training=True)

            # calculate losses
            gen_loss = self.generator.generator_loss(fake_output)
            disc_loss = self.discriminator.discriminator_loss(real_output, fake_output)

        # calculate gradients and apply them
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
                 
    def restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    