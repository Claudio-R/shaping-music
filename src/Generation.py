from models.gan import GenerativeAdversarialNetwork
import tensorflow as tf

def build_gan():
    print("Defining the GAN")
    gan = GenerativeAdversarialNetwork()
    return gan

def train_gan(audio_urls:list):
    gan = build_gan()
    print("Training the GAN")
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, 100])
    gan.fit(audio_urls, epochs=5, seed=seed)
    gan.save_weights()
    return gan