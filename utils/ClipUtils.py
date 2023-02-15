import os
from moviepy.editor import *

def create_clip(frames_dir, audio_path, fps=2):
    frames_files = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir)]
    frames = sorted(frames_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

    audio_clip = AudioFileClip(audio_path)
    frame_duration = 1 / fps

    clip = concatenate([ImageClip(f).set_duration(frame_duration) for f in frames], method="compose")

    if audio_clip.duration > clip.duration:
        audio_clip = audio_clip.subclip(0, clip.duration)
    elif audio_clip.duration < clip.duration:
        audio_clip = audio_clip.fx(vfx.loop, duration=clip.duration)

    clip = clip.set_audio(audio_clip)

    clips_count = len(os.listdir('data/generated_clips'))
    clip.write_videofile(f"data/generated_clips/clip_{clips_count}.mp4", fps=fps)

def retrieve_frames(path_dir):
    frames_list = []
    for filename in os.listdir(path_dir):
        path = os.path.join(path_dir, filename)
        frames_list.append(path)
    return frames_list