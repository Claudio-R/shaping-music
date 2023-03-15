import tensorflow as tf
from typing import Dict

class Discriminator(tf.Module):
    def __init__(self, input_shapes:list):
        '''
        Discriminator model for the GAN:
        - input_shapes: list of input shapes for each audio embedding
        returns: Discriminator model
        '''
        self.input_shapes = tf.reduce_prod(input_shapes, axis=1)

        input_layers = [tf.keras.layers.Input(shape=(shape, ), dtype=tf.float32, name=f'input_audio_embedding_{i}') for i, shape in enumerate(self.input_shapes)]
        dense_1 = [tf.keras.layers.Dense(shape, name=f'dense1_embedding_{i}' )(input_layers[i]) for i, shape in enumerate(self.input_shapes)]
        leaky_relu_1 = [tf.keras.layers.LeakyReLU(name=f'leaky_relu1_embedding_{i}')(dense) for i, dense in enumerate(dense_1)]
        dropout_1 = [tf.keras.layers.Dropout(0.3, name=f'dropout1_embedding_{i}')(leaky_relu) for i, leaky_relu in enumerate(leaky_relu_1)]

        dense_2 = [tf.keras.layers.Dense(128, name=f'dense2_embedding_{i}')(dropout_1[i]) for i, dropout in enumerate(dropout_1)]
        leaky_relu_2 = [tf.keras.layers.LeakyReLU(name=f'leaky_relu2_embedding_{i}')(dense) for i, dense in enumerate(dense_2)]
        dropout_2 = [tf.keras.layers.Dropout(0.3, name=f'dropout2_embedding_{i}')(leaky_relu) for i, leaky_relu in enumerate(leaky_relu_2)]

        concat = tf.keras.layers.Concatenate(name='concat')(dropout_2)
        dense_3 = tf.keras.layers.Dense(1, name='dense3')(concat)
        output = tf.keras.layers.Activation('sigmoid', name='output')(dense_3)
        
        self.model = tf.keras.Model(inputs=input_layers, outputs=output, name='discriminator')
        self.model.compile(optimizer='adam', loss=self.discriminator_loss)
        tf.keras.utils.plot_model(self.model, to_file='data/debug/discriminator.png', show_shapes=True, show_layer_names=True)

    def __call__(self, audio_embeds:list, training:bool=True):
        '''
        Call the discriminator model:
        - audio_embeds: list of audio embeddings
        '''
        self.__validate_input(audio_embeds)
        audio_embeds = self.__prepare_input(audio_embeds)
        return self.model(audio_embeds, training=training)
    
    @staticmethod
    def discriminator_loss(real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def __validate_input(self, audio_embeds:list):
        assert len(audio_embeds[0]) == len(self.input_shapes), '''
        Number of audio embeddings must be equal to the number of input shapes.
        Shapes: {} != {}
        '''.format(len(audio_embeds[0]), len(self.input_shapes))
        for i in range(len(audio_embeds[0])):
            assert tf.reduce_prod(audio_embeds[0][i].shape) == self.input_shapes[i], '''
            Audio embedding shape must be equal to the output shape.
            Check the output shape at index {}.
            Shapes: {} != {}
            '''.format(i, audio_embeds[0][i].shape, self.output_shapes[i])
        return

    @staticmethod
    def __prepare_input(audio_embeds:list) -> Dict[str, tf.Tensor]:
        '''
        Flatten the audio embeddings, than convert them to dictionaries of tensors
        '''
        for i in range(len(audio_embeds)):
            for j in range(len(audio_embeds[i])):
                audio_embeds[i][j] = tf.reshape(audio_embeds[i][j], [-1])
            audio_embeds[i] = [ audio_embeds[i][j] for j in range(len(audio_embeds[i])) ]
    
        # Transpose the video and audio embeddings, than convert them to dictionaries of tensors
        audio_embeds = [ [ audio_embeds[i][j] for i in range(len(audio_embeds)) ] for j in range(len(audio_embeds[0])) ]
        audio_embeds = { f'input_audio_embedding_{i}': audio_embeds[i] for i in range(len(audio_embeds)) }
        audio_embeds = { k: tf.convert_to_tensor(v) for k, v in audio_embeds.items() }
        # print('Audio:', audio_embeds.keys())
        return audio_embeds
    
    def save_weights(self, path:str):
        self.model.save_weights(path)

if __name__ == "__main__":
    discriminator = Discriminator([(1, 1024), (1, 512)])