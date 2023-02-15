import os, sys
import numpy as np
import tensorflow as tf
import PIL
from models.multimodal_feature_extractor import MultimodalFeatureExtractor
from models.image_to_sound_encoder import ImageToSoundEncoder
from models.sound_to_image_encoder import SoundToImageEncoder
from models.generator import Generator
from models.discriminator import Discriminator
from utils.AudioUtils import split_audio
import librosa

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
            print("Epoch: {}".format(epoch+1))
            for i, audio_url in enumerate(audio_urls):
                self.__training_step(audio_url)
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('='*int((i+1)/len(audio_urls)*20), int((i+1)/len(audio_urls)*100)))
                sys.stdout.flush()
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*20, 100))
            sys.stdout.flush()
            print()
                
            if (epoch + 1) % 5 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
            
            print("Generating and saving images")
            img = self.generator(seed, training=False)
            img = (img[0, :, :, :] * 127.5 + 127.5).numpy().astype(np.uint8)
            PIL.Image.fromarray(img).save('data/generated_images/image_at_epoch_{:04d}.png'.format(epoch))
                
        self.generator.save_weights('data/weights/gan_generator.h5')
        self.discriminator.save_weights('data/weights/gan_discriminator.h5')
        
    def __training_step(self, audio_url):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_tape.watch(self.generator.trainable_variables)
            disc_tape.watch(self.discriminator.trainable_variables)

            # 1. generate an image from the original audio
            input_wavs = self.generator.preprocess(audio_url)
            generated_image = self.generator(input_wavs, training=True)
            generated_image = tf.expand_dims(generated_image[0, :, :, :], 0)

            # 2. extract image embeddings from generated image
            generated_image_embeds = self.mfe.predict_from_image(generated_image)
            generated_image_embeds = tf.expand_dims(generated_image_embeds, 0)

            # 3. encode image embeddings to sound embeddings
            generated_audio_embeds = self.i2sEncoder(generated_image_embeds, training=False)

            # 4. extract audio embeddings from original audio
            original_audio_embeds = self.mfe('sound', audio_url)

            # 5. feed audio features to discriminator
            real_output = self.discriminator(original_audio_embeds, training=True)
            fake_output = self.discriminator(generated_audio_embeds, training=True)

            # 6. calculate losses
            gen_loss = self.generator.generator_loss(fake_output)
            disc_loss = self.discriminator.discriminator_loss(real_output, fake_output)

        # 7. calculate gradients and apply them
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
                 
    def restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    def save_weights(self):
        self.generator.save_weights('data/weights/gan_generator.h5')
        self.discriminator.save_weights('data/weights/gan_discriminator.h5')
    
    def load_weights(self):
        self.generator.load_weights('data/weights/gan_generator.h5')
        self.discriminator.load_weights('data/weights/gan_discriminator.h5')

    def create_clip(self, song_path, sound_dir, fps=2):
        frames_dir = 'data/generated_images'
        
        # 1. clear the frames directory and sound directory
        for f in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, f))
        for f in os.listdir(sound_dir):
            os.remove(os.path.join(sound_dir, f))
        
        # 2. estimate BPM of the song and calculate the frame rate
        y, sr = librosa.load(song_path)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(tempo)
        fps = bpm / 60
        print("Estimated BPM: {}".format(bpm))
        print("Frame rate: {}".format(fps))
        
        # 3. split the audio into frames
        split_audio(song_path, sound_dir, fps)
        
        # 4. generate images from the frames
        sound_urls = [os.path.join(sound_dir, f) for f in os.listdir(sound_dir)]
        images = []
        for i, sound_url in enumerate(sound_urls):
            input_wavs = self.generator.preprocess(sound_url)
            generated_image = self.generator(input_wavs, training=False)
            generated_image = (generated_image[0, :, :, :] * 127.5 + 127.5).numpy().astype(np.uint8)
            images.append(generated_image)
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*int((i+1)/len(sound_urls)*20), int((i+1)/len(sound_urls)*100)))
            sys.stdout.flush()
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*20, 100))
        sys.stdout.flush()
        print()

        # 5. save the images
        for i in range(len(images)):
            PIL.Image.fromarray(images[i]).save('data/generated_images/image{}.png'.format(i))


        return frames_dir, song_path