from pydub import AudioSegment
import os

def split_audio(audio_file, dir, fps):
    audio = AudioSegment.from_wav(audio_file)
    duration = audio.duration_seconds
    sampling_period = 1 / fps
    num_segments = int(duration / sampling_period)

    for i in range(num_segments):
        start = i * sampling_period * 1000
        end = (i + 1) * sampling_period * 1000
        newAudio = audio[start:end]
        newAudio.export(os.path.join(dir, f"audioSegment_{i}.wav"), format="wav")