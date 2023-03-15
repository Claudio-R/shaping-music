from models.gan import GenerativeAdversarialNetwork
import tensorflow as tf

def train_gan(audio_urls:list, video_embeds_shapes, audio_embeds_shapes):
    gan = GenerativeAdversarialNetwork(video_embeds_shapes, audio_embeds_shapes)
    num_examples_to_generate = 1
    seed = tf.random.normal([num_examples_to_generate, 100])
    gan.train(seed, audio_urls)
    return gan