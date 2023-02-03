import tensorflow as tf
from yamnet.inference import YamNet
from utils.AudioUtils import load_wav

class SoundModel(tf.keras.Model):
    '''
    YAMNet-based model for embedding sounds
    '''
    def __init__(self, *args, **kwargs):
        yamnet = YamNet()
        yamnet.trainable = False
        output_layers = ['layer14/pointwise_conv/relu']
        output_layers = [yamnet.get_layer(name).output for name in output_layers]
        yamnet_output = output_layers[-1]
        yamnet_output_shape = yamnet_output.shape
        pool_size = (min(yamnet_output_shape[1], yamnet_output_shape[2]))
        pooling_layer = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=None)(yamnet_output)
        output_layer = tf.keras.layers.Flatten()(pooling_layer)
        super(SoundModel, self).__init__([yamnet.input], output_layer, name='SoundEmbeddingsExtractor', *args, **kwargs)

    def __call__(self, wav_url:str):
        waveform = self.__preprocess(wav_url)
        return super(SoundModel, self).__call__(waveform)

    def __preprocess(self, wav_url:str):
        waveform = load_wav(wav_url)
        waveform = waveform / tf.int16.max
        return waveform

    def get_dataset(self, wav_urls:list):
        '''
        Returns a tf.data.Dataset containing the embeddings of the sounds in wav_urls
        '''
        waveforms = [self.__preprocess(url) for url in wav_urls]
        return tf.data.Dataset.from_tensor_slices(waveforms).map(lambda x: self(x))

if __name__ == '__main__':
    sound_model = SoundModel()
    sound_model.summary()