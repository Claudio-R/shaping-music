from utils.ClipUtils import create_clip
from utils.VideoUtils import preprocess_video
from utils.AudioUtils import split_audio
from models.gan import GenerativeAdversarialNetwork

def create_clip_from_url(video_url, image_dir, sound_dir, fps=2):
    frames_path, audio_path = preprocess_video(video_url, image_dir, sound_dir, fps)
    create_clip(frames_path, audio_path, fps)

def create_videoclip(song_path, image_dir, sound_dir, fps=2):
    print(split_audio(song_path, sound_dir, fps))
    gan = GenerativeAdversarialNetwork()
    gan.load_weights()
    gan.generate_images()

    create_clip(frames_path, audio_path, fps)