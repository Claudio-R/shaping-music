import tensorflow as tf
import tensorflow_hub as hub
import librosa
import soundfile as sf

class SoundModel():
    '''
    A class that extracts features from a wav file and returns the embeddings for the audio content.
    '''
    def __init__(self, *args, **kwargs):
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')

    def __call__(self, wav_url:str) -> tf.Tensor:
        waveform = self.__preprocess(wav_url)
        _, embeddings, _ = self.model(waveform)
        return embeddings
    
    def __preprocess(self, wav_url:str) -> tf.Tensor:
        # 1. Load the WAV file, resample it to 16 kHz, and convert it to mono
        wav, sr = librosa.load(wav_url, sr=16000, mono=True)
        # 2. Save the wav file for debugging
        sf.write('data/debug/test.wav', wav, sr)
        return tf.convert_to_tensor(wav, dtype=tf.float32)

if __name__ == '__main__':
    sound_model = SoundModel()
    embeds = sound_model('data/input_sounds/segment_2.wav')
    print("Output shape:", embeds.shape)
    print("Output:", embeds)