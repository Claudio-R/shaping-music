import tensorflow as tf

class Discriminator(tf.keras.Sequential):
    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        # First Layer upsample to 14x14
        self.add(tf.keras.layers.Input(shape=(1024)))
        self.add(tf.keras.layers.Reshape((4, 4, 64)))

        self.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        self.add(tf.keras.layers.LeakyReLU())
        self.add(tf.keras.layers.Dropout(0.3))

        # Second Layer
        self.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.add(tf.keras.layers.LeakyReLU())
        self.add(tf.keras.layers.Dropout(0.3))

        # Third Layer
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(1))

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss