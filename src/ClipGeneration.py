from utils.VideoUtils import create_clip
from models.gan import GenerativeAdversarialNetwork

def create_videoclip(gan:GenerativeAdversarialNetwork, song_url:str, fps:int):
    _, _ = gan(song_url, fps)