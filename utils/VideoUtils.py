import os
import numpy as np
from moviepy.editor import *
import imageio
import librosa
import soundfile as sf

def preprocess_video(video_url:str, desired_fps:int=2, output_ext="wav") -> tuple:
    '''Extracts frames from video and audio from video and saves them to disk'''

    # 0. Get filename
    pathname, _ = os.path.splitext(video_url)
    filename = pathname.split('/')[-1]

    # 1. Clear directories
    frames_dir = 'data/input_frames'
    sounds_dir = 'data/input_sounds'
    [os.remove(os.path.join(frames_dir, f)) for f in os.listdir(frames_dir)] if os.path.exists(frames_dir) else os.mkdir(frames_dir)
    [os.remove(os.path.join(sounds_dir, f)) for f in os.listdir(sounds_dir)] if os.path.exists(sounds_dir) else os.mkdir(sounds_dir)
        
    # 2. Extract audio content
    videoclip = VideoFileClip(video_url)
    audioClip = videoclip.audio
    audioFile = f"data/audio/{filename}.{output_ext}"
    audioClip.write_audiofile(audioFile, fps=44100, nbytes=2, buffersize=200000)
    audioClip.close()
   
    # 3. Extract frames and audio segments
    original_fps = videoclip.fps
    video_duration = videoclip.duration
    new_fps = min(original_fps, desired_fps)
    sampling_period = 1 / new_fps 
    intervals = [i for i in np.arange(0, video_duration, sampling_period)]
 
    for i, t_start in enumerate(intervals[:-1]):
        # Use imageio to save frames
        frame = videoclip.get_frame(t_start)
        imageio.imwrite(os.path.join(frames_dir, f"frame_{i}.jpg"), frame)

        # Use librosa (moviepy has a bug)
        duration = intervals[i+1] - t_start
        y, sr = librosa.load(audioFile, sr=44100, offset=t_start, duration=duration)
        sf.write(os.path.join(sounds_dir, f"segment_{i}.wav"), y, sr)
    
    # 4. Collect frames and audio segments in lists and return
    frames = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir)]
    sounds = [os.path.join(sounds_dir, f) for f in os.listdir(sounds_dir)]
    frames.sort(key= lambda x: int(''.join(filter(str.isdigit, x))))
    sounds.sort(key= lambda x: int(''.join(filter(str.isdigit, x))))
    assert len(frames) == len(sounds)

    return frames, sounds, audioFile

    