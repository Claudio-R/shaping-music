import numpy as np
import scipy 
from scipy.io import wavfile
from pydub import AudioSegment
import os

def load_wav(wav_url:str, desired_sample_rate=16000):
    sample_rate, waveform = wavfile.read(wav_url, 'rb')

    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    
    if sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    
    return waveform

def get_audio_segment(start, end, audio_file, dir, count):
    start_ms = start * 1000; 
    end_ms = end * 1000; 
    name = audio_file + ".wav" 
    newAudio = AudioSegment.from_wav(name)
    newAudio = newAudio[start_ms:end_ms]
    newAudio.export(os.path.join(dir, f"audioSegment_{count}.wav"), format="wav")

def split_audio(audio_file, dir, fps):
    audio = AudioSegment.from_wav(audio_file)
    duration = audio.duration_seconds
    count = 0
    for i in range(0, int(duration), int(1/fps)):
        get_audio_segment(i, i+1, audio_file, dir, count)
        count += 1
    
    return count