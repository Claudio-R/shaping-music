import tensorflow as tf
import numpy as np

class SoundToImageEncoder(tf.keras.Sequential):
    def __init__(self, path_to_embeddings:str='data/embeddings/embeds.npz', *args, **kwargs):
        embeddings = np.load(path_to_embeddings)
        sound_embeds = embeddings['audio_embeds']
        image_embeds = embeddings['video_embeds']
        input_shape = sound_embeds.shape[1:] # (1, 1024)
        output_shape = image_embeds.shape[-1] # (1, 25088)

        super(SoundToImageEncoder, self).__init__([
            tf.keras.layers.Input(shape=input_shape, dtype=tf.float32, name='input_sound_embedding'),
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.LSTM(2048, return_sequences=True),
            tf.keras.layers.Dense(output_shape, name='output_image_embedding'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Softmax()
            ])
        self.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, x, y, *args, **kwargs):
        normalization = tf.keras.layers.Normalization(axis=None, mean=0, variance=1, name='normalization')
        y = normalization(y) 
        super(SoundToImageEncoder, self).fit(x, y, *args, **kwargs)

if __name__ == "__main__":
    encoder = SoundToImageEncoder()
    encoder.summary()
    print(encoder.output_shape)

