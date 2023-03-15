from typing import Tuple, List
import tensorflow as tf
class ImageToSoundEncoder(tf.Module):
    def __init__ (self, input_shapes:list, output_shapes:list):
        self.input_shapes = tf.reduce_prod(input_shapes, axis=1)
        self.output_shapes = tf.reduce_prod(output_shapes, axis=1)

        input_layers = [tf.keras.layers.Input(shape=(shape, ), dtype=tf.float32, name=f'input_image_embedding_{i}') for i, shape in enumerate(self.input_shapes)]
        dense_1 = [tf.keras.layers.Dense(512, activation='relu', name=f'dense1_embedding_{i}')(input_layers[i]) for i, _ in enumerate(self.input_shapes)]
        dense_2 = [tf.keras.layers.Dense(256, activation='relu', name=f'dense2_embedding_{i}')(dense_1[i]) for i, _ in enumerate(self.input_shapes)]
        concat_layer = tf.keras.layers.Concatenate(name='concat_image_embeddings')(dense_2)
        output_layers = [tf.keras.layers.Dense(shape, activation='relu', name=f'output_sound_embedding_{i}')(concat_layer) for i, shape in enumerate(self.output_shapes)]

        self.model = tf.keras.Model(inputs=input_layers, outputs=output_layers)
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        tf.keras.utils.plot_model(self.model, to_file='data/debug/i2sEncoder.png', show_shapes=True, show_layer_names=True)

    def __call__(self, video_embeds:list, *args, **kwargs) -> List[tf.Tensor]:
        self.__validate_input(video_embeds, None)
        video_embeds, _ = self.__prepare_input(video_embeds, None)
        return self.model(video_embeds, *args, **kwargs)

    # @tf.function
    def fit(self, video_embeds:list, audio_embeds:list, epochs:int=5):
        print('\nTraining the Image to Sound Encoder:')
        self.__validate_input(video_embeds, audio_embeds)
        video_embeds, audio_embeds = self.__prepare_input(video_embeds, audio_embeds)
        self.model.fit(video_embeds, audio_embeds, epochs=epochs)
        self.model.save_weights('data/weights/image_to_sound_encoder.h5')

    def __validate_input(self, video_embeds:list, audio_embeds:list):
        # assert len(video_embeds) == len(audio_embeds), 'Number of video and audio embeddings must be equal.'
        if video_embeds:
            assert len(video_embeds[0]) == len(self.input_shapes), 'Number of video embeddings must be equal to the number of input shapes.'
            for i in range(len(video_embeds[0])): 
                assert tf.reduce_prod(video_embeds[0][i].shape) == self.input_shapes[i], '''
                Video embedding shape must be equal to the input shape.
                Check the input shape at index {}.
                Shapes: {} != {}
                '''.format(i, video_embeds[0][i].shape, self.input_shapes[i])
        
        if audio_embeds:
            assert len(audio_embeds[0]) == len(self.output_shapes), 'Number of audio embeddings must be equal to the number of output shapes.'
            for i in range(len(audio_embeds[0])):
                assert tf.reduce_prod(audio_embeds[0][i].shape) == self.output_shapes[i], '''
                Audio embedding shape must be equal to the output shape.
                Check the output shape at index {}.
                Shapes: {} != {}
                '''.format(i, audio_embeds[0][i].shape, self.output_shapes[i])

    @staticmethod
    def __prepare_input(video_embeds:list, audio_embeds:list) -> Tuple[dict, dict]:
        '''
        Flatten the video and audio embeddings, than convert them to dictionaries of tensors
        '''
        if video_embeds:
            for i in range(len(video_embeds)):
                for j in range(len(video_embeds[i])):
                    video_embeds[i][j] = tf.reshape(video_embeds[i][j], [-1])
            video_embeds = [ [ video_embeds[i][j] for i in range(len(video_embeds)) ] for j in range(len(video_embeds[0])) ]
            video_embeds = { f'input_image_embedding_{i}': video_embeds[i] for i in range(len(video_embeds)) }
            video_embeds = { k: tf.convert_to_tensor(v) for k, v in video_embeds.items() }

        if audio_embeds:
            for i in range(len(audio_embeds)):
                for j in range(len(audio_embeds[i])):
                    audio_embeds[i][j] = tf.reshape(audio_embeds[i][j], [-1])
                audio_embeds[i] = [ audio_embeds[i][j] for j in range(len(audio_embeds[i])) ]
            audio_embeds = [ [ audio_embeds[i][j] for i in range(len(audio_embeds)) ] for j in range(len(audio_embeds[0])) ]
            audio_embeds = { f'output_sound_embedding_{i}': audio_embeds[i] for i in range(len(audio_embeds)) }
            audio_embeds = { k: tf.convert_to_tensor(v) for k, v in audio_embeds.items() }

        return video_embeds, audio_embeds
    
    def get_shapes(self) -> Tuple[list, list]:
        input_shapes = [tf.expand_dims(shape, axis=0) for shape in self.input_shapes]
        output_shapes = [tf.expand_dims(shape, axis=0) for shape in self.output_shapes]
        return input_shapes, output_shapes

        

if __name__ == "__main__":
    encoder = ImageToSoundEncoder([(1, 64, 64), (1, 128, 128)], [(1, 512), (1, 1048)])