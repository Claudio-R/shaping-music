import os
import numpy as np
from moviepy.editor import *
import imageio
import librosa
import soundfile as sf

def preprocess_video(video_url:str, desired_fps:int=2, output_ext="wav") -> tuple:
    '''Extracts frames from video and audio from video and saves them to disk'''

    # 0. Get filename
    if not os.path.exists(video_url): raise Exception("Video was not found!")
    pathname, _ = os.path.splitext(video_url)
    filename = pathname.split('/')[-1]

    # 1. Clear or create directories
    images_dir = 'data/mfe/images'
    audios_dir = 'data/mfe/audios'
    videos_dir = 'data/mfe/video'

    if not os.path.exists(images_dir): os.makedirs(images_dir)
    if not os.path.exists(audios_dir): os.makedirs(audios_dir)
    if not os.path.exists(videos_dir): os.makedirs(videos_dir)
    
    [os.remove(os.path.join(images_dir, f)) for f in os.listdir(images_dir)] if os.path.exists(images_dir) else os.mkdir(images_dir)
    [os.remove(os.path.join(audios_dir, f)) for f in os.listdir(audios_dir)] if os.path.exists(audios_dir) else os.mkdir(audios_dir)
        
    # 2. Extract audio content
    videoclip = VideoFileClip(video_url)
    audioClip = videoclip.audio
    audioFile = f"{videos_dir}/{filename}.{output_ext}"
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
        imageio.imwrite(os.path.join(images_dir, f"frame_{i}.jpg"), frame)

        # Use librosa (moviepy has a bug)
        duration = intervals[i+1] - t_start
        y, sr = librosa.load(audioFile, sr=44100, offset=t_start, duration=duration)
        sf.write(os.path.join(audios_dir, f"segment_{i}.wav"), y, sr)
    
    # 4. Collect frames and audio segments in lists and return
    frames = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
    sounds = [os.path.join(audios_dir, f) for f in os.listdir(audios_dir)]
    frames.sort(key= lambda x: int(''.join(filter(str.isdigit, x))))
    sounds.sort(key= lambda x: int(''.join(filter(str.isdigit, x))))
    assert len(frames) == len(sounds)

    return frames, sounds, audioFile

def create_clip(frames_dir, song_url, clips_dir='data/debug/clips'):

    print('''Creating clip from frames in {} and audio in {}'''.format(frames_dir, song_url))

    if not os.path.exists(path=frames_dir): os.makedirs(frames_dir)
    if not os.path.exists(path=clips_dir): os.makedirs(clips_dir)
    if not os.path.exists(path=song_url): raise Exception("The song was not found!")

    frames_files = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir)]
    frames = sorted(frames_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

    audio_clip = AudioFileClip(song_url)
    song_duration = audio_clip.duration
    frame_duration = song_duration / len(frames)
    fps = 1 / frame_duration

    clip = concatenate([ImageClip(f).set_duration(frame_duration) for f in frames], method="compose")

    if audio_clip.duration > clip.duration:
        audio_clip = audio_clip.subclip(0, clip.duration)
    elif audio_clip.duration < clip.duration:
        audio_clip = audio_clip.fx(vfx.loop, duration=clip.duration)

    clip = clip.set_audio(audio_clip)

    clips_count = len(os.listdir(clips_dir))
    clip.write_videofile(os.path.join(clips_dir, 'generated_clip_{}.mp4'.format(clips_count)), fps=fps)

if __name__ == "__main__":
    frames_dir = 'data/gan/images'
    audio_path = 'data/test_video/test1.wav'
    create_clip(frames_dir, audio_path)