import numpy as np
from models.image_to_sound_encoder import ImageToSoundEncoder
from models.sound_to_image_encoder import SoundToImageEncoder

# def encode_data(path_to_embeddings:str='data/embeddings/embeds.npz'):
def encode_data(data:dict):
    # s2iEncoder = SoundToImageEncoder(data)
    i2sEncoder = ImageToSoundEncoder(data)
    i2sEncoder.fit(data, epochs=5)

    # embeds = np.load(path_to_embeddings)
    # sound_embeds = embeds['audio_embeds']
    # image_embeds = embeds['video_embeds']

    # print("\nTraining the Sound to Image Encoder")
    # s2iEncoder.fit(sound_embeds, image_embeds, epochs=5)
    # s2iEncoder.save_weights('data/weights/sound_to_image_encoder.h5')
    
    # print("\nTraining the Image to Sound Encoder")
    # i2sEncoder.fit(image_embeds, sound_embeds, epochs=5)
    # i2sEncoder.save_weights('data/weights/image_to_sound_encoder.h5')

    return i2sEncoder