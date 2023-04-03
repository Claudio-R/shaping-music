import tensorflow as tf

class ConvolutionalAutoencoder(tf.keras.Model):
    # def __init__(self):
    #     super(ConvolutionalAutoencoder, self).__init__()
    #     self.encoder = tf.keras.Sequential([
    #         tf.keras.layers.Input(shape=(28, 28, 1)),
    #         tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
    #         tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

    #     self.decoder = tf.keras.Sequential([
    #         tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
    #         tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
    #         tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    # def call(self, x):
    #     encoded = self.encoder(x)
    #     decoded = self.decoder(encoded)
    #     return decoded

    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        self.style_extractor = self.define_style_extractor()
        self.encoder = self.define_encoder()
        print(self.encoder.summary())
        self.decoder = self.define_decoder()
        self.img_SIZE = 28
        
    def call(self, img):
        # img = self.preprocess(img_url)
        # convert to rgb
        img = tf.image.grayscale_to_rgb(img)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        embeds = self.style_extractor(img)
        styles = self.get_style(embeds)
        encoded = self.encoder(styles)
        decoded_img = self.decoder(encoded)
        return decoded_img

    def define_style_extractor(self):
        vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        styles = ['block1_conv1', 'block2_conv1', 'block3_conv1']#, 'block4_conv1', 'block5_conv1']
        styles_layers = [vgg19.get_layer(style).output for style in styles]
        return tf.keras.Model(inputs=vgg19.input, outputs=styles_layers, name='style_extractor', trainable=False)

    def define_encoder(self):
        '''
        multiple inputs but single output
        '''
        self.img_shapes = self.style_extractor.output_shape
        self.latent_dim = 100

        input_layers = [tf.keras.layers.Input(shape=(img_shape[1], img_shape[2], img_shape[3]), dtype=tf.float32, name=f'input_{i}') for i, img_shape in enumerate(self.img_shapes)]
        conv2d_layers_1 = [tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(input_layer) for input_layer in input_layers]
        conv2d_layers_2 = [tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(conv2d_layer_1) for conv2d_layer_1 in conv2d_layers_1]
        dense_layers = [tf.keras.layers.Dense(self.latent_dim, activation='relu')(conv2d_layer_2) for conv2d_layer_2 in conv2d_layers_2]
        first_layer_shape = tf.shape(dense_layers[0])
        resizing_layers = [tf.image.resize(dense_layer, (first_layer_shape[1], first_layer_shape[2])) for dense_layer in dense_layers]
        concat_layer = tf.keras.layers.Concatenate(axis=-1)(resizing_layers)
        output_layer = tf.keras.layers.Dense(self.latent_dim, activation='relu')(concat_layer)
        return tf.keras.Model(inputs=input_layers, outputs=output_layer, name='encoder')

    def define_decoder(self):
        '''
        single input, single output
        '''
        input_layer = tf.keras.layers.Input(shape=(None, None, self.latent_dim), dtype=tf.float32, name='latent_input')
        conv2dTranspose_layer_1 = tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same')(input_layer)
        conv2dTranspose_layer_2 = tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same')(conv2dTranspose_layer_1)
        conv2d_layer = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(conv2dTranspose_layer_2)
        return tf.keras.Model(inputs=input_layer, outputs=conv2d_layer, name='decoder')
    
    def preprocess(self, img_url:str) -> tf.Tensor:
        img = tf.keras.preprocessing.image.load_img(img_url, target_size=(self.img_SIZE, self.img_SIZE))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.expand_dims(img, axis=0)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img

    def get_style(self, embeds:list) -> tf.Tensor:
        for embed in embeds:
            if tf.rank(embed) == 3:
                embed = tf.expand_dims(embed, 0)
            result = tf.linalg.einsum('bijc,bijd->bcd', embed, embed)
            input_shape = tf.shape(embed)
            num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
            embed = (result / num_locations)
        return embeds
    
if __name__ == '__main__':
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    autoencoder = ConvolutionalAutoencoder()
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    autoencoder.fit(x_train, x_train,
                    epochs=10,
                    shuffle=True,
                    validation_data=(x_test, x_test))
        
    n = 10
    x_test = x_test[:n]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(autoencoder(x_test)[i].numpy().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('data/debug/convolutional_autoencoder.png')
