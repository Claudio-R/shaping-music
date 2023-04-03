from models.image_to_sound_encoder import ImageToSoundEncoder
from models.sound2image_autoencoder import Sound2ImageAutoencoder

def encode_data(data:dict, epochs:int=1):
    '''
    Trains the encoders and saves the weights.
    '''
    video_embeds = data['video_embeds']
    audio_embeds = data['audio_embeds']
    input_shapes = [embed.shape for embed in video_embeds[0]]
    output_shapes = [embed.shape for embed in audio_embeds[0]]
    # i2sEncoder = ImageToSoundEncoder(input_shapes, output_shapes)
    s2iAutoencoder = Sound2ImageAutoencoder(input_shapes, output_shapes)
    s2iAutoencoder.train(video_embeds, audio_embeds, epochs=epochs)

    # i2sEncoder.fit(video_embeds, audio_embeds, epochs=epochs)

    return s2iAutoencoder