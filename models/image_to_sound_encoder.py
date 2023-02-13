import numpy as np
import tensorflow as tf

class ImageToSoundEncoder(tf.keras.Sequential):
    def __init__ (self, path_to_embeddings:str='data/processed/embeddings.npz'):
        embeddings = np.load(path_to_embeddings)
        image_embeds = embeddings['video_embeds']
        sound_embeds = embeddings['audio_embeds']
        input_shape = (None, image_embeds.shape[-1])
        output_shape = sound_embeds.shape[-1]

        # TODO: more weights ends up exhausting GPU memory
        super(ImageToSoundEncoder, self).__init__([
            tf.keras.layers.Input(shape=input_shape, dtype=tf.float32, name='input_image_embedding'),
            tf.keras.layers.LSTM(512, return_sequences=True),
            tf.keras.layers.Dense(output_shape, name='output_sound_embedding'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Softmax()
            ])
        self.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        def fit(self, x, y, *args, **kwargs):
            normalization = tf.keras.layers.Normalization(axis=None, mean=0, variance=1, name='normalization')
            y = normalization(y)
            super(ImageToSoundEncoder, self).fit(x, y, *args, **kwargs)

if __name__ == "__main__":
    encoder = ImageToSoundEncoder()
    encoder.summary()
    print(encoder.output_shape)