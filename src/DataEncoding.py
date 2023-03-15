import numpy as np
from models.image_to_sound_encoder import ImageToSoundEncoder
from models.sound_to_image_encoder import SoundToImageEncoder

# def encode_data(path_to_embeddings:str='data/embeddings/embeds.npz'):
def encode_data(data:dict):
    '''
    Trains the encoders and saves the weights.
    '''
    video_embeds = data['video_embeds']
    audio_embeds = data['audio_embeds']
    input_shapes = [embed.shape for embed in video_embeds[0]]
    output_shapes = [embed.shape for embed in audio_embeds[0]]
    i2sEncoder = ImageToSoundEncoder(input_shapes, output_shapes)
    i2sEncoder.fit(video_embeds, audio_embeds, epochs=2)

    return i2sEncoder