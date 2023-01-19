import librosa

def split_audio(file_name):
    audio, FS = librosa.load(file_name)
    Ts = 0.5
    hop_size = int(Ts*FS)
    N = int(len(audio)/hop_size)
    audio_list = [ audio[i*hop_size:(i+1)*hop_size] for i in range(N)]
    return audio_list