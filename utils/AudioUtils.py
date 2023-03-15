from pydub import AudioSegment
import os

def split_audio(audio_url:str, fps:int):
    audio = AudioSegment.from_wav(audio_url)
    duration = audio.duration_seconds
    sampling_period = 1 / fps
    num_segments = int(duration / sampling_period)

    # create a temporary directory to store the audio segments

    for i in range(num_segments):
        start = i * sampling_period * 1000
        end = (i + 1) * sampling_period * 1000
        newAudio = audio[start:end]
        newAudio.export(os.path.join(dir, f"audioSegment_{i}.wav"), format="wav")