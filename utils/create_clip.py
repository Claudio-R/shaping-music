import os
import moviepy
from moviepy.editor import VideoFileClip, AudioFileClip

def create_clip(images_path, audio_path):
    generated_frames_list = []
    for filename in os.listdir(images_path):
        path = os.path.join(images_path, filename)
        # path = os.path.join(os.getcwd(), path)
        generated_frames_list.append(path)

    # response = requests.get(image_url)
    # image = Image.open(BytesIO(response.content))
    # clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(generated_frames_list, fps)
    # audio_clip = moviepy.audio.io.AudioFileClip.AudioFileClip(audio_filename)
    
    # clip.set_audio(audio_clip);
    # clip.write_videofile(video_filename)
    # np_image = np.array(image)

    fps = 2
    audio_filename = audio_path
    video_filename = 'data/generated_clip/Test.mp4'

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(generated_frames_list, fps)
    clip.write_videofile(video_filename)

    audio_clip = AudioFileClip(audio_filename)
    # audio_clip = audio_clip.subclip(0, 25)

    video_clip = VideoFileClip(video_filename);
    video_clip = video_clip.set_audio(audio_clip);

    video_clip.write_videofile(video_filename)