import numpy as np
import tensorflow as tf

class ImageToSoundEncoder(tf.keras.Sequential):
    def __init__ (self, path_to_embeddings:str='data/processed/embeddings.npz'):
        embeddings = np.load(path_to_embeddings)
        image_embeds = embeddings['video_embeds']
        sound_embeds = embeddings['audio_embeds']
        input_shape = image_embeds.shape[1:]
        output_shape = sound_embeds.shape[1:]

        super(ImageToSoundEncoder, self).__init__([
            tf.keras.layers.Input(shape=(input_shape)),
            tf.keras.layers.ConvLSTM2D(input_shape[-1], kernel_size=(2,2), activation='relu'),
            tf.keras.layers.Conv2D(output_shape[-1], kernel_size=(2,2)),
            tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=None),
        ])
        self.compile(optimizer='adam', loss='binary_crossentropy')