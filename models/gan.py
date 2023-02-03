import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython import display
import PIL
from models.multimodal_feature_extractor import MultimodalFeatureExtractor
from models.image_to_sound_encoder import ImageToSoundEncoder
from models.generator import Generator
from models.discriminator import Discriminator

class GenerativeAdversarialNetwork():
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.feature_extractor = MultimodalFeatureExtractor()
        self.image_to_sound_encoder = ImageToSoundEncoder()
        self.image_to_sound_encoder.load_weights('models/weights/image_to_sound_encoder.h5')

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.checkpoint_dir = 'data/checkpoints/gan_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)
    
    def fit(self, audio_urls, epochs, seed):
        for epoch in range(epochs):
            start = time.time()
            for audio_batch in audio_urls:
                self.__train_step(audio_batch)
            display.clear_output(wait=True)
            self.generate_and_save_images(self.generator, epoch + 1, seed)
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        display.clear_output(wait=True)
        self.__generate_and_save_images(self.generator, epochs, seed)
        
    @tf.function
    def __train_step(self, audio_urls, BATCH_SIZE=32, noise_dim=100):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # generate images from noise
            generated_image = self.generator(noise, training=True)
            print(generated_image.shape) # (32, 64, 64, 3) wrong shape
            
            # encode images to audio
            generated_audio_embeds = self.image_to_sound_encoder(generated_image, training=True)
            
            # extract audio features from audio urls
            original_audio_embeds = self.feature_extractor('sound', audio_urls, training=True)
            
            # feed audio features to discriminator
            real_output = self.discriminator(original_audio_embeds, training=True)
            fake_output = self.discriminator(generated_audio_embeds, training=True)
            
            # calculate losses
            gen_loss = self.generator.generator_loss(fake_output)
            disc_loss = self.discriminator.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
                 
    def __generate_and_save_images(self, model, epoch, test_input, seed):
        size = int(np.sqrt(seed.shape[0]))
        predictions = model(test_input, training=False)
        fig = plt.figure(figsize=(size*2, size*2))
        for i in range(predictions.shape[0]):
            plt.subplot(size, size, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

    def restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def display_image(self, epoch_no):
        return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

    def save_weights(self):
        self.generator.save_weights('models/weights/generator.h5')
        self.discriminator.save_weights('models/weights/discriminator.h5')
    

if __name__ == '__main__':
    gan = GenerativeAdversarialNetwork(Generator, Discriminator)
    