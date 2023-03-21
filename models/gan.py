import os
import numpy as np
import tensorflow as tf
import PIL
from models.multimodal_feature_extractor import MultimodalFeatureExtractor
from models.image_to_sound_encoder import ImageToSoundEncoder
from models.generator import Generator
from models.discriminator import Discriminator
from utils.OutputUtils import print_progress_bar
from utils.VideoUtils import create_clip
import librosa, soundfile as sf

class GenerativeAdversarialNetwork(tf.Module):
    def __init__(self, video_embeds_shapes:list, audio_embeds_shapes:list):
        self.generator = Generator()
        self.discriminator = Discriminator(audio_embeds_shapes)
        self.mfe = MultimodalFeatureExtractor()

        self.i2sEncoder = ImageToSoundEncoder(video_embeds_shapes, audio_embeds_shapes)
        try:
            self.i2sEncoder.model.load_weights('data/weights/image_to_sound_encoder.h5')
        except:
            print("\n### Error in loading i2sEncoder weights ###\n")

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
        
    def __call__(self, song_url:str, fps:int):
        '''
        Generate a new video clip from a given song
        '''
        print('\nCreating clip for a new song:')
        audios_dir = 'data/gan/audios'
        images_dir = 'data/gan/images'
        clip_dir = 'data/gan/videos'

        if not os.path.exists(audios_dir): os.makedirs(audios_dir)
        if not os.path.exists(images_dir): os.makedirs(images_dir)
        if not os.path.exists(clip_dir): os.makedirs(clip_dir)

        # 0. Clear the directories
        for dir in [audios_dir, images_dir]:
            for file in os.listdir(dir):
                os.remove(os.path.join(dir, file))

        # 1. Split audio into segments
        wav, sr = librosa.load(song_url, sr=44100)
        song_duration = librosa.get_duration(y=wav, sr=sr)
        sampling_period = 1 / fps 
        intervals = [i for i in np.arange(0, song_duration, sampling_period)]
        for i, t_start in enumerate(intervals[:-1]):
            duration = intervals[i+1] - t_start
            y, sr = librosa.load(song_url, sr=44100, offset=t_start, duration=duration)
            sf.write(os.path.join(audios_dir, f"segment_{i}.wav"), y, sr)

        # 2. Generate images for each segment and save them
        audio_urls = [os.path.join(audios_dir, f"segment_{i}.wav") for i in range(len(intervals[:-1]))]
        for i, audio_url in enumerate(audio_urls):
            input_wavs = self.generator.preprocess(audio_url)
            generated_image = self.generator(input_wavs, training=False)
            generated_image = (generated_image[0, :, :, :] * 127.5 + 127.5).numpy().astype(np.uint8)
            PIL.Image.fromarray(generated_image).save(os.path.join(images_dir, f"image_{i}.jpg"))
            print_progress_bar(i+1, len(audio_urls), prefix='Generating frames:', length=50, fill='=')

        # 3. Create and save the clip
        create_clip(images_dir, song_url, 'data/gan/videos')
        return images_dir, clip_dir
    
    def train(self, seed, audio_urls:list, epochs:int=2):
        print('\nTraining the Gan:')
        for epoch in range(epochs):
            for i, audio_url in enumerate(audio_urls):
                if not os.path.exists(audio_url): continue
                self.__training_step(audio_url)
                print_progress_bar(i+1, len(audio_urls), prefix='Epoch: {}/{}'.format(epoch+1, epochs), length=50, fill='=')
                
            if (epoch + 1) % 5 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
            
            img = self.generator(seed, training=False)
            img = (img[0, :, :, :] * 127.5 + 127.5).numpy().astype(np.uint8)
            PIL.Image.fromarray(img).save('data/debug/generated_training_images/training_image_at_epoch_{:04d}.jpg'.format(epoch))
                
        self.generator.save_weights('data/weights/gan_generator.h5')
        self.discriminator.save_weights('data/weights/gan_discriminator.h5')
        
    def __training_step(self, audio_url):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_tape.watch(self.generator.trainable_variables)
            disc_tape.watch(self.discriminator.trainable_variables)

            # 1. generate an image from the original audio
            input_wavs = self.generator.preprocess(audio_url)
            generated_image = self.generator(input_wavs, training=True)
            generated_image = generated_image[0, :, :, :]

            # 2. extract image embeddings from generated image
            generated_image_embeds = self.mfe.predict_from_image(generated_image)

            # 3. encode image embeddings to sound embeddings
            generated_audio_embeds = self.i2sEncoder([generated_image_embeds], training=False)
            if type(generated_audio_embeds) != list: generated_audio_embeds = [[generated_audio_embeds]]

            # 4. extract audio embeddings from original audio
            original_audio_embeds = self.mfe.predict_from_file(audio_url, verbose=False)

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

    def load_weights(self):
        self.generator.load_weights('data/weights/gan_generator.h5')
        self.discriminator.load_weights('data/weights/gan_discriminator.h5')