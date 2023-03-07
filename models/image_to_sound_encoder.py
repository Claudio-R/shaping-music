import tensorflow as tf

class ImageToSoundEncoder():
    def __init__ (self, data:dict):
        self.data = data
        image_embeds = data['video_embeds']
        sound_embeds = data['audio_embeds']

        flatten = tf.keras.layers.Flatten()
        for i in range(len(image_embeds)):
            for j in range(len(image_embeds[i])):
                image_embeds[i][j] = flatten(image_embeds[i][j])

        input_shapes = [embed.shape for embed in image_embeds[0]]
        output_shape = sound_embeds[0].shape[-1]
        
        input_layer = [tf.keras.layers.Input(shape=shape, dtype=tf.float32, name=f'input_image_embedding_{i}') for i, shape in enumerate(input_shapes)]
        concat_layer = tf.keras.layers.Concatenate(name='concat_image_embeddings')(input_layer)
        output_layer = tf.keras.layers.Dense(output_shape, activation='relu', name='output_sound_embedding')(concat_layer)
        
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    def __call__(self, image_embeds:list) -> tf.Tensor:
        image_embeds = self.data['video_embeds'] if image_embeds is None else image_embeds
        return self.model(image_embeds)

    def fit(self, data:dict, epochs:int=2, batch_size:int=1):
        image_embeds = data['video_embeds']
        sound_embeds = data['audio_embeds']

        flatten = tf.keras.layers.Flatten()
        for i in range(len(image_embeds)):
            for j in range(len(image_embeds[i])):
                image_embeds[i][j] = flatten(image_embeds[i][j])

        for i in range(len(image_embeds)):
            # expand dims to add batch dimension
            image_embeds[i] = [tf.expand_dims(embed, axis=0) for embed in image_embeds[i]]
            self.model.fit(image_embeds[i], sound_embeds, epochs=epochs, batch_size=batch_size)

        tf.keras.utils.plot_model(self.model, to_file='data/debug/i2sEncoder.png', show_shapes=True, show_layer_names=True)
        self.save_weights('data/weights/image_to_sound_encoder.h5')

if __name__ == "__main__":
    encoder = ImageToSoundEncoder()
    encoder.summary()
    tf.keras.utils.plot_model(encoder, to_file='data/debug/i2sEncoder.png', show_shapes=True, show_layer_names=True)