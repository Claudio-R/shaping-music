import numpy as np
import librosa
import scipy 
from scipy.io import wavfile

def load_wav(wav_url:str, desired_sample_rate=16000):
    sample_rate, waveform = wavfile.read(wav_url, 'rb')

    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    
    if sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    
    return waveform

def split_audio(file_name):
    audio, FS = librosa.load(file_name)
    Ts = 0.5
    hop_size = int(Ts*FS)
    N = int(len(audio)/hop_size)
    audio_list = [ audio[i*hop_size:(i+1)*hop_size] for i in range(N)]
    return audio_list