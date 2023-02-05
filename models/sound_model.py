import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

class SoundModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        super(SoundModel, self).__init__(*args, **kwargs)

    def __call__(self, wav_url:str) -> tf.Tensor:
        waveform = self.__preprocess(wav_url)
        scores, embeddings, spectrogram = self.model(waveform)
        return embeddings
    
    def __preprocess(self, wav_url:str) -> tf.Tensor:
        """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
        file_contents = tf.io.read_file(wav_url)
        wav, sample_rate = tf.audio.decode_wav(
            file_contents,
            desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
        return wav

if __name__ == '__main__':
    sound_model = SoundModel()
    print(sound_model('data/test_samples/segment2.wav'))