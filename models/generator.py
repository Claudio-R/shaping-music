from typing import Any
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

SIZE = 64 # this should be 512 for the final model

class Generator(tf.keras.Sequential):
    def __init__(self, input_dim=100, SIZE=64, *args, **kwargs):
 
        self.input_dim = input_dim
            
        super(Generator, self).__init__(*args, **kwargs)
        self.add(tf.keras.layers.Dense(SIZE//4*SIZE//4*256, use_bias=False, input_shape=(self.input_dim,)))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.LeakyReLU())

        self.add(tf.keras.layers.Reshape((SIZE//4, SIZE//4, 256)))

        self.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.LeakyReLU())

        self.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.LeakyReLU())

        self.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.LeakyReLU())

        assert self.output_shape == (None, SIZE, SIZE, 3)
    
    def __call__(self, input_wavs:tf.Tensor, *args, **kwargs) -> Any:
        return super().__call__(input_wavs, *args, **kwargs)

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    
    def preprocess(self, wav_url:str) -> tf.Tensor:
        """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
        sample_rate, wav = wavfile.read(wav_url)
        wav = wav.astype('float32')
        # convert to mono
        if len(wav.shape) > 1:
            wav = np.mean(wav, axis=1)
        # resample to 16 kHz
        if sample_rate != 16000:
            wav = signal.resample(wav, int(16000 * len(wav) / sample_rate))
        waveform = tf.convert_to_tensor(wav, dtype=tf.float32)

        # split waveform into batches
        num_batches = waveform.shape[0] // self.input_dim
        input_wavs = tf.reshape(waveform[:num_batches*self.input_dim], (num_batches, self.input_dim))
        return input_wavs
    

if __name__ == "__main__":
    generator = Generator()
    generator.summary()
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    print(generated_image.shape)