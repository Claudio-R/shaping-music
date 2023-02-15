import cv2 
import os
import numpy as np
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
# import time
from utils.AudioUtils import get_audio_segment

def extract_audio(video_file, output_ext="wav"):
    '''Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood'''
    filename, _ = os.path.splitext(video_file)
    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(f"{filename}.{output_ext}", 44100, 2, 2000, "pcm_s16le")
    in_name = filename + "." + output_ext
    out_name = 'data/test_audio/test.wav'
    sound = AudioSegment.from_mp3(in_name)
    sound.export(out_name, format="wav")
    return filename, out_name

def preprocess_video(video_url:str, image_dir:str, sound_dir:str, desired_fps:int=2):
    if not os.path.isdir(image_dir): os.mkdir(image_dir)
    if not os.path.isdir(sound_dir): os.mkdir(sound_dir)

    for f in os.listdir(image_dir): os.remove(os.path.join(image_dir, f))
    for f in os.listdir(sound_dir): os.remove(os.path.join(sound_dir, f))

    audio_file_name, out_name = extract_audio(video_url)
   
    video_cap = cv2.VideoCapture(video_url)
    original_fps = video_cap.get(cv2.CAP_PROP_FPS)

    video_duration = video_cap.get(cv2.CAP_PROP_FRAME_COUNT) / original_fps
    new_fps = min(original_fps, desired_fps)
    sampling_period = 1 / new_fps 
    clips_startTime = [i for i in np.arange(0, video_duration, sampling_period)]

    global previous_moment  
    previous_moment = 0

    for i, clip_startTime in enumerate(clips_startTime):
        video_cap.set(cv2.CAP_PROP_POS_MSEC, clip_startTime*1000)
        is_read, frame = video_cap.read()
        if not is_read: break
        path = os.path.join(image_dir, f"frame_{i}.jpg")
        cv2.imwrite(path, frame)

        if clip_startTime - previous_moment >= sampling_period:
            get_audio_segment(previous_moment, clip_startTime, audio_file_name, sound_dir, i)
        previous_moment = clip_startTime
    
    video_cap.release()
    cv2.destroyAllWindows()
    return image_dir, out_name