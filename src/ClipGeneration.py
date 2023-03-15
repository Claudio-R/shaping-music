from utils.ClipUtils import create_clip
from utils.VideoUtils import preprocess_video
from models.gan import GenerativeAdversarialNetwork
import tensorflow as tf

def create_clip_from_url(video_url, image_dir, sound_dir, fps=2):
    frames_dir, audio_path = preprocess_video(video_url, image_dir, sound_dir, fps)
    create_clip(frames_dir, audio_path, fps)

def create_videoclip(gan:GenerativeAdversarialNetwork, song_url, fps=2):
    gan.load_weights()
    images_dir, clip_dir = gan(song_url, fps)
    create_clip(images_dir, song_url, fps, clip_dir)