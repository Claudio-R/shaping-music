# NOTE: mnist images are normalized in the range [0, 1]
# TODO: define a custom loop

import tensorflow as tf
import tqdm

class ConvolutionalAutoencoder(tf.keras.Model):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        self.img_SIZE = 28
        self.style_extractor = self.define_style_extractor()
        self.encoder = self.define_encoder()
        self.decoder = self.define_decoder()

        # plot the models
        tf.keras.utils.plot_model(self.style_extractor, to_file='style_extractor.png', show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(self.encoder, to_file='encoder.png', show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(self.decoder, to_file='decoder.png', show_shapes=True, show_layer_names=True)
        
    def train(self, training_data, validation_data, epochs=10):
        self.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        for e in range(epochs):
            print(f'Epoch {e+1}/{epochs}')
            loss = 0
            for batch in tqdm.tqdm(training_data, desc='Training'):
                loss += self.train_step(batch)
            loss /= len(training_data)
            print(f'Loss: {loss}')
            loss = 0
            for batch in tqdm.tqdm(validation_data, desc='Validation'):
                loss += self.test_step(batch)
            loss /= len(validation_data)
            print(f'Validation Loss: {loss}')
            
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            decoded_img = self.call(batch)
            loss = self.loss(decoded_img, batch)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
    
    def test_step(self, batch):
        decoded_img = self.call(batch)
        loss = self.loss(decoded_img, batch)
        return loss

    def call(self, imgs):
        # img = self.preprocess(img_url)
        # convert to rgb
        imgs = tf.image.grayscale_to_rgb(imgs)
        imgs = tf.keras.applications.vgg19.preprocess_input(imgs)
        embeds = self.style_extractor(imgs)
        styles = self.get_style(embeds)
        encoded_imgs = self.encoder(styles)
        decoded_imgs = self.decoder(encoded_imgs)
        return decoded_imgs

    def define_style_extractor(self):
        vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        styles = ['block1_conv1', 'block2_conv1', 'block3_conv1']#, 'block4_conv1', 'block5_conv1']
        styles_layers = [vgg19.get_layer(style).output for style in styles]
        return tf.keras.Model(inputs=vgg19.input, outputs=styles_layers, name='style_extractor', trainable=False)

    def define_encoder(self) -> tf.keras.Model:
        '''
        multiple inputs but single output model.
        - input: the style embeddings (None, SIZE, SIZE, CHANNELS)
        - output: the encoded image (None, 1024)
        '''
        self.img_shapes = self.style_extractor.output_shape
        if type(self.img_shapes) == tuple:
            self.img_shapes = [self.img_shapes]
        self.latent_dim = 1024

        input_layers = [tf.keras.layers.Input(shape=(img_shape[1], img_shape[2], img_shape[3]), dtype=tf.float32, name=f'input_{i}') for i, img_shape in enumerate(self.img_shapes)]
        conv2d_layers_1 = [tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2, name=f'encoder_conv2d_1_{i}')(input_layer) for i, input_layer in enumerate(input_layers)]
        conv2d_layers_2 = [tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2, name=f'encoder_conv2d_2_{i}')(conv2d_layer_1) for i, conv2d_layer_1 in enumerate(conv2d_layers_1)]
        dense_layers_1 = [tf.keras.layers.Dense(self.latent_dim, activation='relu', name=f'encoder_dense_1_{i}')(conv2d_layer_2) for i, conv2d_layer_2 in enumerate(conv2d_layers_2)]
        pooling_layers = [tf.keras.layers.GlobalAveragePooling2D()(dense_layer) for dense_layer in dense_layers_1]
        concatenate_layer = tf.keras.layers.Concatenate(axis=-1)(pooling_layers)
        dense_layer_2 = tf.keras.layers.Dense(self.latent_dim, activation='relu', name=f'encoder_dense_2')(concatenate_layer)
        return tf.keras.Model(inputs=input_layers, outputs=dense_layer_2, name='encoder')

    def define_decoder(self) -> tf.keras.Model:
        '''
        single input, single output model.
        - input: the encoded image (None, 1024)
        - output: the reconstructed image (None, SIZE, SIZE, 3)
        '''
        self.NUM_CHANNELS = 1
        input_layer = tf.keras.layers.Input(shape=(self.latent_dim), dtype=tf.float32, name='latent_input')
        dense_layer_1 = tf.keras.layers.Dense(self.latent_dim, activation='relu', name='decoder_dense_1')(input_layer)
        expand_layer = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, axis=1), axis=1))(dense_layer_1) # (None, 1, 1, 1024)
        conv2dTranspose_layer_1 = tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same', name='decoder_conv2d_transpose_1')(expand_layer) # (None, 8, 8, 8)
        conv2dTranspose_layer_2 = tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same', name='decoder_conv2d_transpose_2')(conv2dTranspose_layer_1) # (None, 16, 16, 16)
        dense_layer_2 = tf.keras.layers.Dense(self.img_SIZE * self.img_SIZE * self.NUM_CHANNELS, activation='relu', name='decoder_dense_2')(conv2dTranspose_layer_2) # (None, 16, 16, 784)
        pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(dense_layer_2) # (None, 784)
        reshape_layer = tf.keras.layers.Reshape((self.img_SIZE, self.img_SIZE, self.NUM_CHANNELS))(pooling_layer) # (None, 28, 28, 1)

        return tf.keras.Model(inputs=input_layer, outputs=reshape_layer, name='decoder')
    
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

    dataset_size = len(x_train)
    x_train = tf.data.Dataset.from_tensor_slices(x_train).shuffle(dataset_size)
    x_train, x_val = x_train.take(int(dataset_size * 0.8)), x_train.skip(int(dataset_size * 0.8))
    x_train = x_train.batch(32).shuffle(1000)
    x_val = x_val.batch(32).shuffle(1000)
    x_test = tf.data.Dataset.from_tensor_slices(x_test).batch(32).shuffle(1000)

    autoencoder = ConvolutionalAutoencoder()
    autoencoder.train(x_train, x_val, epochs=1)
    # autoencoder.fit(x_train, x_train,
    #                 epochs=1,
    #                 shuffle=True,
    #                 validation_data=(x_test, x_test))

        
    n = 10
    x_test = x_test.take(n).as_numpy_iterator().next()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        img = x_test[i].reshape(28, 28)
        plt.imshow(img)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)

        img = autoencoder(tf.expand_dims(x_test[i], axis=0)).numpy().reshape(28, 28)
        plt.imshow(img)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
   
    plt.savefig('data/debug/convolutional_autoencoder.png')
    print('Done!')
 