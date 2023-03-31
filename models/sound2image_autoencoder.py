from typing import Tuple
import tensorflow as tf
import os
from utils.OutputUtils import print_progress_bar

class Sound2ImageAutoencoder(tf.Module):
    def __init__ (self, img_shapes:list, snd_shapes:list):
        debug_dir = 'data/debug'
        weights_dir = 'data/weights'
        if not os.path.exists(debug_dir): os.makedirs(debug_dir)
        if not os.path.exists(weights_dir): os.makedirs(weights_dir)

        self.img_shapes = tf.reduce_prod(img_shapes, axis=1)
        self.snd_shapes = tf.reduce_prod(snd_shapes, axis=1)

        self.encoder = self.__define_encoder()
        self.decoder = self.__define_decoder()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        
        # self.autoencoder = tf.keras.Model(inputs=self.encoder.inputs, outputs=self.decoder(self.encoder.outputs))
        # self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        # tf.keras.utils.plot_model(self.autoencoder, to_file='data/debug/autoencoder.png', show_shapes=True, show_layer_names=True)

    def __define_encoder(self) -> tf.keras.Model:
        input_layers = [tf.keras.layers.Input(shape=(shape, ), dtype=tf.float32, name=f'input_image_embedding_{i}') for i, shape in enumerate(self.img_shapes)]
        dense_1 = [tf.keras.layers.Dense(512, activation='relu', name=f'encoded_dense1_embedding_{i}')(input_layers[i]) for i, _ in enumerate(self.img_shapes)]
        dense_2 = [tf.keras.layers.Dense(256, activation='relu', name=f'encoded_dense2_embedding_{i}')(dense_1[i]) for i, _ in enumerate(self.img_shapes)]
        concat_layer = tf.keras.layers.Concatenate(name='concat_image_embeddings')(dense_2)
        output_layers = [tf.keras.layers.Dense(shape, activation='relu', name=f'encoded_sound_embedding_{i}')(concat_layer) for i, shape in enumerate(self.snd_shapes)]
        return tf.keras.Model(inputs=input_layers, outputs=output_layers, name='encoder')
    
    def __define_decoder(self) -> tf.keras.Model:
        input_layers = [tf.keras.layers.Input(shape=(shape, ), dtype=tf.float32, name=f'input_sound_embedding_{i}') for i, shape in enumerate(self.snd_shapes)]
        dense_1 = [tf.keras.layers.Dense(256, activation='relu', name=f'decoded_dense1_embedding_{i}')(input_layers[i]) for i, _ in enumerate(self.snd_shapes)]
        dense_2 = [tf.keras.layers.Dense(512, activation='relu', name=f'decoded_dense2_embedding_{i}')(dense_1[i]) for i, _ in enumerate(self.snd_shapes)]
        concat_layer = tf.keras.layers.Concatenate(name='concat_sound_embeddings')(dense_2)
        output_layers = [tf.keras.layers.Dense(shape, activation='relu', name=f'decoded_image_embedding_{i}')(concat_layer) for i, shape in enumerate(self.img_shapes)]
        return tf.keras.Model(inputs=input_layers, outputs=output_layers, name='decoder')

    @tf.function
    def train(self, video_embeds:list, audio_embeds:list, epochs:int=5) -> None:
        print('\nTraining the Image to Sound Encoder:')
        video_embeds, audio_embeds = self.__prepare_input(video_embeds, audio_embeds)
        for epoch in range(epochs):
            for i, (img_embeds, snd_embeds) in enumerate(zip(video_embeds, audio_embeds)):
                loss = self.training_step(img_embeds, snd_embeds)
                print_progress_bar(i, len(video_embeds), prefix='Epoch: {}/{}'.format(epoch, epochs, loss.numpy()), suffix='Loss: {}'.format(loss.numpy()), length=50, fill='=')

    def training_step(self, img:dict, snd:dict) -> tf.Tensor:
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
            encoded = self.encoder(img)
            decoded = self.decoder(encoded)
            loss = self.loss(encoded, decoded, img, snd)
            
        encoder_gradients = enc_tape.gradient(loss, self.encoder.trainable_variables)
        decoder_gradients = dec_tape.gradient(loss, self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_variables))
        self.optimizer.apply_gradients(zip(decoder_gradients, self.decoder.trainable_variables))
        return loss

    def loss(self, encoded:list, decoded:list, imgs:list, snds:list) -> tf.Tensor:
        reconstruction_loss = tf.reduce_mean([tf.reduce_mean(tf.square(decoded[i] - imgs[i])) for i, _ in enumerate(imgs)])
        latent_loss = tf.reduce_mean([tf.reduce_mean(tf.square(snds[i] - encoded[i])) for i, _ in enumerate(snds)])
        return reconstruction_loss + latent_loss

    # def __call__(self, video_embeds:list, *args, **kwargs) -> List[tf.Tensor]:
    #     video_embeds, _ = self.__prepare_input(video_embeds, None)
    #     return self.autoencoder(video_embeds, *args, **kwargs)

    # # @tf.function
    # def fit(self, video_embeds:list, audio_embeds:list, epochs:int=5):
    #     print('\nTraining the Image to Sound Encoder:')
    #     video_embeds, audio_embeds = self.__prepare_input(video_embeds, audio_embeds)
    #     self.autoencoder.fit(video_embeds, video_embeds, epochs=epochs)
    #     self.autoencoder.save_weights('data/weights/s2i_autoencoder.h5')

    def __validate_input(self, video_embeds:list, audio_embeds:list):
        if video_embeds:
            assert len(video_embeds[0]) == len(self.img_shapes), 'Number of video embeddings must be equal to the number of input shapes.'
            for i in range(len(video_embeds[0])): 
                assert tf.reduce_prod(video_embeds[0][i].shape) == self.img_shapes[i], '''
                Video embedding shape must be equal to the input shape.
                Check the input shape at index {}.
                Shapes: {} != {}
                '''.format(i, video_embeds[0][i].shape, self.img_shapes[i])
        
        if audio_embeds:
            assert len(audio_embeds[0]) == len(self.snd_shapes), 'Number of audio embeddings must be equal to the number of output shapes.'
            for i in range(len(audio_embeds[0])):
                assert tf.reduce_prod(audio_embeds[0][i].shape) == self.snd_shapes[i], '''
                Audio embedding shape must be equal to the output shape.
                Check the output shape at index {}.
                Shapes: {} != {}
                '''.format(i, audio_embeds[0][i].shape, self.snd_shapes[i])

    def __prepare_input(self, video_embeds:list, audio_embeds:list) -> Tuple[dict, dict]:
        '''
        Flatten the video and audio embeddings, then convert them to dictionaries of tensors
        Example:
            (
                {
                    'image_embedding_0': [tf.Tensor, tf.Tensor, ...],
                    'image_embedding_1': [tf.Tensor, tf.Tensor, ...],
                    ...
                },
                {
                    'sound_embedding_0': [tf.Tensor, tf.Tensor, ...],
                    'sound_embedding_1': [tf.Tensor, tf.Tensor, ...],
                    ...
                }
            )    
            
        '''

        self.__validate_input(video_embeds, audio_embeds)

        if video_embeds:
            for i in range(len(video_embeds)):
                for j in range(len(video_embeds[i])):
                    video_embeds[i][j] = tf.reshape(video_embeds[i][j], [-1])
            video_embeds = [ [ video_embeds[i][j] for i in range(len(video_embeds)) ] for j in range(len(video_embeds[0])) ]
            video_embeds = { f'image_embedding_{i}': video_embeds[i] for i in range(len(video_embeds)) }
            video_embeds = { k: tf.convert_to_tensor(v) for k, v in video_embeds.items() }

        if audio_embeds:
            for i in range(len(audio_embeds)):
                for j in range(len(audio_embeds[i])):
                    audio_embeds[i][j] = tf.reshape(audio_embeds[i][j], [-1])
                audio_embeds[i] = [ audio_embeds[i][j] for j in range(len(audio_embeds[i])) ]
            audio_embeds = [ [ audio_embeds[i][j] for i in range(len(audio_embeds)) ] for j in range(len(audio_embeds[0])) ]
            audio_embeds = { f'sound_embedding_{i}': audio_embeds[i] for i in range(len(audio_embeds)) }
            audio_embeds = { k: tf.convert_to_tensor(v) for k, v in audio_embeds.items() }

        return video_embeds, audio_embeds
        

if __name__ == "__main__":
    encoder = Sound2ImageAutoencoder([(1, 64, 64), (1, 128, 128)], [(1, 512), (1, 1048)])