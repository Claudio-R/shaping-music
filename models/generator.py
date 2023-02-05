import tensorflow as tf
import matplotlib.pyplot as plt

class Generator(tf.keras.Sequential):
    def __init__(self, BATCH_SIZE=1, input_dim=100, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)
        # First Layer
        self.add(tf.keras.layers.Input(shape=(input_dim,)))
        self.add(tf.keras.layers.Dense(7*7*BATCH_SIZE, use_bias=False))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.LeakyReLU())
        self.add(tf.keras.layers.Reshape((7, 7, BATCH_SIZE)))

        # Second Layer
        self.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.LeakyReLU())

        # Third Layer
        self.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.LeakyReLU())

        # Fourth Layer downsample to 7x7
        self.add(tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.LeakyReLU())

        # Fifth Layer
        self.add(tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)


if __name__ == "__main__":
    generator = Generator()
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    print(generated_image.shape)