from utils.ClipUtils import create_clip
from utils.VideoUtils import preprocess_video
from models.gan import GenerativeAdversarialNetwork

def create_clip_from_url(video_url, image_dir, sound_dir, fps=2):
    frames_dir, audio_path = preprocess_video(video_url, image_dir, sound_dir, fps)
    create_clip(frames_dir, audio_path, fps)

def create_videoclip(song_path, sound_dir, fps=2):
    gan = GenerativeAdversarialNetwork()
    gan.restore()
    frames_dir, song_path = gan.create_clip(song_path, sound_dir, fps)
    create_clip(frames_dir, song_path, fps)