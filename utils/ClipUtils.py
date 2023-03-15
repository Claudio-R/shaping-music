import os
from moviepy.editor import *

def create_clip(frames_dir, song_url, clips_dir='data/debug/clips'):
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

def retrieve_frames(path_dir):
    frames_list = []
    for filename in os.listdir(path_dir):
        path = os.path.join(path_dir, filename)
        frames_list.append(path)
    return frames_list

if __name__ == "__main__":
    frames_dir = 'data/gan/images'
    audio_path = 'data/test_video/test1.wav'
    create_clip(frames_dir, audio_path)