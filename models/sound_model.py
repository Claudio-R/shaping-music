import tensorflow as tf
import tensorflow_hub as hub
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import numpy as np

class SoundModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        super(SoundModel, self).__init__(*args, **kwargs)

    def __call__(self, wav_url:str) -> tf.Tensor:
        waveform = self.__preprocess(wav_url)
        _, embeddings, _ = self.model(waveform)
        return embeddings
    
    def __preprocess(self, wav_url:str) -> tf.Tensor:
        """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
        sample_rate, wav = wavfile.read(wav_url)
        wav = wav.astype('float32')
        # convert to mono
        if len(wav.shape) > 1:
            wav = np.mean(wav, axis=1)
        # resample to 16 kHz
        if sample_rate != 16000:
            wav = signal.resample(wav, int(16000 * len(wav) / sample_rate))
        return tf.convert_to_tensor(wav, dtype=tf.float32)

if __name__ == '__main__':
    sound_model = SoundModel()
    embeds = sound_model('data/test_samples/segment2.wav')
    print("Output shape:", embeds.shape)
    print("Output type:", embeds.dtype)