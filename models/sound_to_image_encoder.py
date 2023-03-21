import tensorflow as tf
import os

class SoundToImageEncoder(tf.keras.Sequential):
    def __init__(self, data:dict, *args, **kwargs):
        sound_embeds = data['audio_embeds']
        image_embeds = data['video_embeds']

        flatten = tf.keras.layers.Flatten()
        for i in range(len(image_embeds)):
            for j in range(len(image_embeds[i])):
                image_embeds[i][j] = flatten(image_embeds[i][j])

        input_shape = sound_embeds[0].shape
        output_shape = [embed.shape for embed in image_embeds[0]]

        print("input_shape: ", input_shape)
        print("output_shape: ", output_shape)

        super(SoundToImageEncoder, self).__init__([
            tf.keras.layers.Input(shape=input_shape, dtype=tf.float32, name='input_sound_embedding'),
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.LSTM(2048, return_sequences=True),
            tf.keras.layers.Dense(output_shape, name='output_image_embedding'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Softmax()
            ], *args, **kwargs)

        self.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.__fit(sound_embeds, image_embeds, epochs=5)

        weights_dir = 'data/weights'
        if not os.path.exists(weights_dir): os.makedirs(weights_dir)

        self.save_weights('data/weights/sound_to_image_encoder.h5')

    def __fit(self, x, y, *args, **kwargs):
        normalization = tf.keras.layers.Normalization(axis=None, mean=0, variance=1, name='normalization')
        y = normalization(y) 
        super(SoundToImageEncoder, self).fit(x, y, *args, **kwargs)

if __name__ == "__main__":
    encoder = SoundToImageEncoder()
    encoder.summary()
    print(encoder.output_shape)

