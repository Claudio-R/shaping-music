from models.gan import GenerativeAdversarialNetwork
import tensorflow as tf

def train_gan(audio_urls:list, video_embeds_shapes, audio_embeds_shapes, SIZE:int=64, epochs:int=1):
    gan = GenerativeAdversarialNetwork(video_embeds_shapes, audio_embeds_shapes, frame_size=SIZE)
    num_examples_to_generate = 1
    seed = tf.random.normal([num_examples_to_generate, 100])
    gan.train(seed, audio_urls, epochs=epochs)
    return gan