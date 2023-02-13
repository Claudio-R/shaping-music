import tensorflow as tf
import matplotlib.pyplot as plt

SIZE = 56 # this should be 512 for the final model

class Generator(tf.keras.Sequential):
    def __init__(self, input_dim=100, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)
        self.add(tf.keras.layers.Dense(SIZE//4*SIZE//4*256, use_bias=False, input_shape=(input_dim,)))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.LeakyReLU())

        self.add(tf.keras.layers.Reshape((SIZE//4, SIZE//4, 256)))

        self.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.LeakyReLU())

        self.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.LeakyReLU())

        self.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.LeakyReLU())

        assert self.output_shape == (None, SIZE, SIZE, 3)
    
    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)


if __name__ == "__main__":
    generator = Generator()
    generator.summary()
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    print(generated_image.shape)