from utils.ClipUtils import create_clip
from utils.VideoUtils import preprocess_video

def create_clip_from_url(video_url, image_dir, sound_dir, fps=2):
    frames_path, audio_path = preprocess_video(video_url, image_dir, sound_dir, fps)
    create_clip(frames_path, audio_path, fps)