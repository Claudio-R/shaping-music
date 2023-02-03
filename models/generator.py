import tensorflow as tf

class Generator(tf.keras.Sequential):
    def __init__(self, BATCH_SIZE=32, input_dim=100, *args, **kwargs):
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
        self.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.LeakyReLU())

        # Fourth Layer
        self.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))        

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)
