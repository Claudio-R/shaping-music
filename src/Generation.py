from models.gan import GenerativeAdversarialNetwork
import tensorflow as tf

def train_gan(audio_urls:list):
    gan = GenerativeAdversarialNetwork()
    num_examples_to_generate = 1
    seed = tf.random.normal([num_examples_to_generate, 100])
    gan.restore()
    gan.train(audio_urls, epochs=50, seed=seed)
    gan.save_weights()
    return gan