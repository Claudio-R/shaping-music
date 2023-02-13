import tensorflow as tf

AUDIO_EMBEDS_SHAPE = 1024 # this should be 512 for the final model

class Discriminator(tf.keras.Sequential):
    def __init__(self, input_shape=(None, 1, AUDIO_EMBEDS_SHAPE), *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.add(tf.keras.layers.Dense(input_shape[1]*input_shape[2]))
        self.add(tf.keras.layers.LeakyReLU())
        self.add(tf.keras.layers.Dropout(0.3))

        self.add(tf.keras.layers.Dense(128))
        self.add(tf.keras.layers.LeakyReLU())
        self.add(tf.keras.layers.Dropout(0.3))

        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(1))

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

if __name__ == "__main__":
    discriminator = Discriminator()
    noise = tf.random.normal([1, 1024])
    discriminator(noise, training=False)
    discriminator.summary()