
import cv2 
import os
import numpy as np
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pytube import YouTube


SAVING_FRAMES_PER_SECOND = 2


def download_from_youtube(link):    
    SAVE_PATH = os.getcwd() 
    yt = YouTube(link)
    yt = yt.get('mp4', '720p')
    yt.download(SAVE_PATH)

# video = cv2.VideoCapture("lana.mp4")
# object representing video

# nr_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = int(video.get(cv2.CAP_PROP_FPS))
# length = nr_frames / fps; 


#download_from_youtube("https://www.youtube.com/watch?v=MVEVhtoYBxE")

# print(f"number of frames ={nr_frames},frames per second ={fps}, video length = {length}")

def convert_video_to_audio_moviepy(video_file, output_ext="wav"):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
    filename, ext = os.path.splitext(video_file)
    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(f"{filename}.{output_ext}", 44100, 2, 2000, "pcm_s16le")
    in_name = filename + "." + output_ext
    out_name = 'data/test_audio/test.wav'
    sound = AudioSegment.from_mp3(in_name)
    sound.export(out_name, format="wav")
    return filename


def split_audio(t1, t2, audio_file, dir, count):
    t1 = t1*1000; 
    t2 = t2*1000; 
    name = audio_file + ".wav" 
    newAudio = AudioSegment.from_wav(name)
    newAudio = newAudio[t1:t2]
    newAudio.export(os.path.join(dir, f"segment{count}.wav"), format="wav")
    

def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s


def preprocess_video(video_url:str, image_dir:str, audio_dir:str, saving_fps:int=SAVING_FRAMES_PER_SECOND):
    global previous_moment  
    previous_moment=0

    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    if not os.path.isdir(audio_dir):
        os.mkdir(audio_dir)

    audio_file_name = convert_video_to_audio_moviepy(video_url)
   
    # read the video file --> returns an object representing the video 
    cap = cv2.VideoCapture(video_url)

    # get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    #We take the minimum between the real fps and the one provided from external
    #we cannot use a fps smaller than the original one 
    saving_frames_per_second = min(fps, saving_fps)

    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)

    count = 0
    generated_frames=0

    while True:
        is_read, frame = cap.read()
        if not is_read:        
            break

        current_time = count/fps
        try:
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break
        
        if closest_duration <= current_time:
            cv2.imwrite(os.path.join(image_dir, f"frame{generated_frames}.jpg"), frame)
            if current_time - previous_moment > 0.2:
                split_audio(previous_moment, current_time, audio_file_name, audio_dir, generated_frames)
            # drop the duration spot from the list, since this duration spot is already saved
            generated_frames += 1
            
            try:
                #if the list is empty 
                previous_moment = current_time
                saving_frames_durations.pop(0)
                
            except IndexError:
                pass

        count += 1