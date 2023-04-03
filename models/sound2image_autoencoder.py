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
        self.encoder_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.decoder_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.checkpoint_dir = 'data/checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "sound2image_ckpt")
        self.checkpoint = tf.train.Checkpoint(
            encoder_optimizer=self.encoder_optimizer,
            decoder_optimizer=self.decoder_optimizer,
            encoder=self.encoder,
            decoder=self.decoder
            )
        
    def __define_encoder(self) -> tf.keras.Model:
        '''
        The encoder should be able to take in a list of images and output a list of embeddings.
        By now, the encoder will be a simple MLP taking in the flattened image and outputting the flattened embedding.
        '''
        input_layers = [tf.keras.layers.Input(shape=(shape, ), dtype=tf.float32, name=f'image_embedding_{i}') for i, shape in enumerate(self.img_shapes)]
        dense_1 = [tf.keras.layers.Dense(512, activation='relu', name=f'encoded_dense1_embedding_{i}')(input_layers[i]) for i, _ in enumerate(self.img_shapes)]
        dense_2 = [tf.keras.layers.Dense(256, activation='relu', name=f'encoded_dense2_embedding_{i}')(dense_1[i]) for i, _ in enumerate(self.img_shapes)]
        concat_layer = tf.keras.layers.Concatenate(name='concat_image_embeddings')(dense_2)
        output_layers = [tf.keras.layers.Dense(shape, activation='relu', name=f'encoded_sound_embedding_{i}')(concat_layer) for i, shape in enumerate(self.snd_shapes)]
        return tf.keras.Model(inputs=input_layers, outputs=output_layers, name='encoder')

    def __define_decoder(self) -> tf.keras.Model:
        '''
        The decoder moves from the latent space (audio embeddings) to the image space.
        It should be able to reconstruct the original image from the audio embeddings (if the input of the encoder is the original image),
        or generate a new image whose style is similar to the original image (if the input of the encoder is the list of image embeddings).
        '''
        input_layers = [tf.keras.layers.Input(shape=(shape, ), dtype=tf.float32, name=f'sound_embedding_{i}') for i, shape in enumerate(self.snd_shapes)]
        dense_1 = [tf.keras.layers.Dense(256, activation='relu', name=f'decoded_dense1_embedding_{i}')(input_layers[i]) for i, _ in enumerate(self.snd_shapes)]
        dense_2 = [tf.keras.layers.Dense(512, activation='relu', name=f'decoded_dense2_embedding_{i}')(dense_1[i]) for i, _ in enumerate(self.snd_shapes)]
        concat_layer = tf.keras.layers.Concatenate(name='concat_sound_embeddings')(dense_2)
        output_layers = [tf.keras.layers.Dense(shape, activation='relu', name=f'decoded_image_embedding_{i}')(concat_layer) for i, shape in enumerate(self.img_shapes)]
        return tf.keras.Model(inputs=input_layers, outputs=output_layers, name='decoder')

    # @tf.function
    def train(self, img_dataset:list, snd_dataset:list, epochs:int=5, batch_size:int=4) -> None:
        print('\nTraining the Image to Sound Encoder:')
        self.encoder_optimizer.build(self.encoder.trainable_variables)
        self.decoder_optimizer.build(self.decoder.trainable_variables)

        img_dataset, snd_dataset = self.__prepare_input(img_dataset, snd_dataset, batch_size)
        for epoch in range(epochs):
            for i, (img_embeds, snd_embeds) in enumerate(zip(img_dataset, snd_dataset)):
                loss = self.training_step(img_embeds, snd_embeds)
                print_progress_bar(i, len(img_dataset), prefix='Epoch: {}/{}'.format(epoch, epochs, loss.numpy()), suffix='Loss: {}'.format(loss.numpy()), length=50, fill='=')
            print('\n')

    def training_step(self, img_embeds:dict, snd_embeds:dict) -> tf.Tensor:
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
            encoded = self.encoder(img_embeds)
            decoded = self.decoder(encoded)
            loss = self.loss(encoded, decoded, img_embeds, snd_embeds)

        encoder_gradients = enc_tape.gradient(loss, self.encoder.trainable_variables)
        decoder_gradients = dec_tape.gradient(loss, self.decoder.trainable_variables)
        self.encoder_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_variables))
        self.decoder_optimizer.apply_gradients(zip(decoder_gradients, self.decoder.trainable_variables))
        return loss

    def loss(self, encoded:list, decoded:list, imgs:dict, snds:dict) -> tf.Tensor:
        imgs = [imgs[key] for key in imgs]
        snds = [snds[key] for key in snds]
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
    def __prepare_input(self, video_embeds:list, audio_embeds:list, batch_size:int) -> Tuple[list, list]:
        '''
        Flatten the video and audio embeddings, then convert them to a list of dictionaries of lists of tensors, whose length is equal to the batch size.
        Example:
            (
                [
                    { 'image_embedding_0': tf.Tensor, 'image_embedding_1': tf.Tensor, ... }, 
                    ... 
                ],
                [
                    { 'sound_embedding_0': tf.Tensor, 'sound_embedding_1': tf.Tensor, ... },
                    ...
                ]
            )

        '''

        self.__validate_input(video_embeds, audio_embeds)

        img_dataset = []
        snd_dataset = []

        if video_embeds:
            # flatten the tensors
            for i in range(len(video_embeds)):
                for j in range(len(video_embeds[i])):
                    video_embeds[i][j] = tf.reshape(video_embeds[i][j], [-1])

            # trim the list to the nearest multiple of the batch size
            video_embeds = video_embeds[:len(video_embeds) - len(video_embeds) % batch_size]

            # convert the list of tensors to a list of dictionaries of lists of tensors
            for i in range(0, len(video_embeds), batch_size):
                img_dataset.append({ 'image_embedding_{}'.format(j): [video_embeds[i+k][j] for k in range(batch_size)]
                    for j in range(len(video_embeds[0])) })
            
            # convert the list of dictionaries of lists of tensors to a list of dictionaries of tensors
            for i in range(len(img_dataset)):
                for k, _ in img_dataset[i].items():
                    img_dataset[i][k] = tf.stack(img_dataset[i][k], axis=0)

        if audio_embeds:
            for i in range(len(audio_embeds)):
                for j in range(len(audio_embeds[i])):
                    audio_embeds[i][j] = tf.reshape(audio_embeds[i][j], [-1])
                
            audio_embeds = audio_embeds[:len(audio_embeds) - len(audio_embeds) % batch_size]

            for i in range(0, len(audio_embeds), batch_size):
                snd_dataset.append({ 'sound_embedding_{}'.format(j): [audio_embeds[i+k][j] for k in range(batch_size)]
                    for j in range(len(audio_embeds[0])) })
                
            for i in range(len(snd_dataset)):
                for k, _ in snd_dataset[i].items():
                    snd_dataset[i][k] = tf.stack(snd_dataset[i][k], axis=0)
            
        return img_dataset, snd_dataset

if __name__ == "__main__":
    encoder = Sound2ImageAutoencoder([(1, 64, 64), (1, 128, 128)], [(1, 512), (1, 1048)])