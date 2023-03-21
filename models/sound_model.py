import tensorflow as tf
import tensorflow_hub as hub
import librosa
from typing import List
from utils.OutputUtils import print_progress_bar
import os

class SoundModel(tf.Module):
    '''
    A class that extracts features from a wav file and returns the embeddings for the audio content.
    '''
    def __init__(self):
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')

    def __call__(self, wav_urls: list, verbose:bool=True) -> List[List[tf.Tensor]]:
        embeddings = []
        for i, wav_url in enumerate(wav_urls):
            if not os.path.exists(wav_url): continue
            waveform = self.__preprocess(wav_url)
            _, embeds, _ = self.model(waveform)
            if type(embeds) != list: embeds = [embeds]
            embeddings.append(embeds)
            if verbose: print_progress_bar(i+1, len(wav_urls), prefix="Extracting audio content:", length = 50, fill = '=')
        return embeddings
    
    def __preprocess(self, wav_url:str) -> tf.Tensor:
        wav, _ = librosa.load(wav_url, sr=16000, mono=True)
        return tf.convert_to_tensor(wav, dtype=tf.float32)

if __name__ == '__main__':
    sound_model = SoundModel()
    embeds = sound_model(['data/input_sounds/segment_2.wav'])