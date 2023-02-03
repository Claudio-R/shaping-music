import numpy as np
from models.image_to_sound_encoder import ImageToSoundEncoder

def build_encoder(path_to_embeddings:str):
    print("Defining the Image to Sound Encoder")
    image_to_sound_encoder = ImageToSoundEncoder(path_to_embeddings)
    return image_to_sound_encoder

def encode_data(path_to_embeddings:str='data/processed/embeddings.npz'):
    encoder = build_encoder(path_to_embeddings)
    
    embeds = np.load(path_to_embeddings)
    image_embeds = embeds['video_embeds']
    sound_embeds = embeds['audio_embeds']

    print("Training the Image to Sound Encoder")
    encoder.fit(image_embeds, sound_embeds, epochs=5)
    encoder.save_weights('models/weights/image_to_sound_encoder.h5')
    
    return encoder