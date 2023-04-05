#TODO: define a custom loss function and a custom training loop to take into account the encoding and decoding losses

import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np
import imageio, librosa, soundfile as sf, moviepy.editor as mp

class ConvolutionalAutoencoder(tf.keras.Model):

    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        self.style_extractor = self.define_style_extractor()
        self.snd_encoder = self.define_sound_encoder()
        self.encoder = self.define_encoder()
        self.decoder = self.define_decoder()
        self.img_SIZE = 28
        
    # def call(self, img) -> tf.Tensor:
    #     # img = self.preprocess(img_url)
    #     # convert to rgb
    #     img = tf.image.grayscale_to_rgb(img)
    #     img = tf.keras.applications.vgg19.preprocess_input(img)
    #     embeds = self.style_extractor(img)
    #     styles = self.get_style(embeds)
    #     encoded = self.encoder(styles)
    #     decoded_img = self.decoder(encoded)
    #     return decoded_img

    def call(self, img_url, wav_url) -> tf.Tensor:
        img = self.preprocess(img_url)
        # convert to rgb
        # img = tf.image.grayscale_to_rgb(img)
        preprocessed_img = tf.keras.applications.vgg19.preprocess_input(img)
        embeds = self.style_extractor(preprocessed_img)
        styles = self.get_style(embeds)
        encoded = self.encoder(styles)
        decoded_img = self.decoder(encoded)
        snd_embeds = self.compute_snd_embeds(wav_url)
        return img, decoded_img, snd_embeds
    
    def compute_loss(self, img, decoded_img, encoded_img, snd_embeds):
        '''
        img: original image
        decoded_img: reconstructed image
        encoded_img: encoded image
        snd_embeds: sound embeddings
        '''
        # compute the reconstruction loss
        reconstruction_loss = tf.reduce_mean(tf.square(decoded_img - img))
        # compute the encoding loss
        print("encoded_img: ", encoded_img.shape)
        print("snd_embeds: ", snd_embeds.shape)
        encoding_loss = tf.reduce_mean(tf.square(encoded_img - snd_embeds))
        # compute the total loss
        total_loss = reconstruction_loss + encoding_loss
        return total_loss
    
    def train_step(self, data:tuple):
        img_url, wav_url = data
        with tf.GradientTape() as tape:
            img, decoded_img, snd_embeds = self(img_url, wav_url)
            loss = self.compute_loss(img, decoded_img, snd_embeds)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

    def train(self, dataset, epochs):
        '''
        dataset: a tf.data.Dataset object containing the images and the corresponding wav files
        epochs: number of epochs to train the model
        
        dataset = tf.data.Dataset.from_tensor_slices((img_urls, wav_urls)) # create a dataset from the image and wav urls, which are stored in lists
        dataset = dataset.map(lambda img_url, wav_url: (img_url, wav_url)) # preprocess the images and return the corresponding wav urls
        dataset = dataset.batch(1) # batch the dataset
        '''
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for img, wav_url in dataset:
                loss = self.train_step((img, wav_url))
                print(f"Loss: {loss}")

    def define_style_extractor(self) -> tf.keras.Model:
        vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        styles = ['block1_conv1', 'block2_conv1', 'block3_conv1']#, 'block4_conv1', 'block5_conv1']
        styles_layers = [vgg19.get_layer(style).output for style in styles]
        return tf.keras.Model(inputs=vgg19.input, outputs=styles_layers, name='style_extractor', trainable=False)

    def define_sound_encoder(self) -> tf.keras.Model:
        return hub.load('https://tfhub.dev/google/yamnet/1')

    def compute_snd_embeds(self, wav_url:str) -> tf.Tensor:
        wav, _ = librosa.load(wav_url, sr=16000, mono=True)
        wav = tf.convert_to_tensor(wav, dtype=tf.float32)
        _, embeds, _ = self.snd_encoder(wav)
        return embeds

    def define_encoder(self) -> tf.keras.Model:
        '''
        multiple inputs but single output
        '''
        self.img_shapes = self.style_extractor.output_shape
        self.latent_dim = 1024 # yamnet embedding size (None, 1024)

        input_layers = [tf.keras.layers.Input(shape=(img_shape[1], img_shape[2], img_shape[3]), dtype=tf.float32, name=f'input_{i}') for i, img_shape in enumerate(self.img_shapes)]
        conv2d_layers_1 = [tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(input_layer) for input_layer in input_layers]
        conv2d_layers_2 = [tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(conv2d_layer_1) for conv2d_layer_1 in conv2d_layers_1]
        dense_layers = [tf.keras.layers.Dense(self.latent_dim, activation='relu')(conv2d_layer_2) for conv2d_layer_2 in conv2d_layers_2]
        desired_shape = tf.shape(dense_layers[0])
        resizing_layers = [tf.image.resize(dense_layer, (desired_shape[1], desired_shape[2])) for dense_layer in dense_layers]
        concat_layer = tf.keras.layers.Concatenate(axis=-1)(resizing_layers)
        output_layer = tf.keras.layers.Dense(self.latent_dim, activation='relu')(concat_layer)
        return tf.keras.Model(inputs=input_layers, outputs=output_layer, name='encoder')

    def define_decoder(self) -> tf.keras.Model:
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

    def create_dataset(self, video_urls:str, FPS:int=12, BATCH_SIZE:int=32) -> tf.data.Dataset:
        '''Extracts frames from video and audio from video and saves them to disk'''

        self.fps = FPS
        self.batch_size = BATCH_SIZE

        img_urls, snd_urls = [], []
        img_idx, snd_idx = 0, 0

        # 0. Clear or create directories
        images_dir = 'data/autoencoder/images'
        audios_dir = 'data/autoencoder/audios'
        music_dir = 'data/autoencoder/music'

        if not os.path.exists(images_dir): os.makedirs(images_dir)
        if not os.path.exists(audios_dir): os.makedirs(audios_dir)
        if not os.path.exists(music_dir): os.makedirs(music_dir)

        [os.remove(os.path.join(images_dir, f)) for f in os.listdir(images_dir)] if os.path.exists(images_dir) else os.mkdir(images_dir)
        [os.remove(os.path.join(audios_dir, f)) for f in os.listdir(audios_dir)] if os.path.exists(audios_dir) else os.mkdir(audios_dir)
        [os.remove(os.path.join(music_dir, f)) for f in os.listdir(music_dir)] if os.path.exists(music_dir) else os.mkdir(music_dir)

        for video_url in video_urls:
            # 1. Get filename
            if not os.path.exists(video_url): raise Exception("Video was not found!")
            pathname, _ = os.path.splitext(video_url)
            filename = pathname.split('/')[-1]

            # 2. Get the video clip
            video = mp.VideoFileClip(video_url)           

            # 3. Extract audio content using moviepy
            music = video.audio
            music_path = os.path.join(music_dir, f"{filename}.wav")
            music.write_audiofile(music_path, fps=44100, nbytes=2, buffersize=200000)
            music.close()

            # 4. Extract frames and audio segments
            video_duration = video.duration
            original_fps = video.fps
            new_fps = min(original_fps, FPS)
            sampling_period = 1 / new_fps
            t_starts = np.arange(0, video_duration, sampling_period)

            for t_start in t_starts[:-1]:
                # Use imageio to save frames
                frame = video.get_frame(t_start)
                frame_path = os.path.join(images_dir, f"frame_{img_idx}.jpg")
                imageio.imwrite(frame_path, frame)
                img_urls.append(frame_path)
                img_idx += 1

                # Use librosa (moviepy has a bug)
                y, sr = librosa.load(music_path, sr=44100, offset=t_start, duration=sampling_period)
                audio_path = os.path.join(audios_dir, f"audio_{snd_idx}.wav")
                sf.write(audio_path, y, sr)    
                snd_urls.append(audio_path)     
                snd_idx += 1
            
            print(f"Video {video_url} was successfully processed!")

        # 5. Create dataset
        assert len(img_urls) == len(snd_urls), "Number of images and audios must be equal!"
        dataset = tf.data.Dataset.from_tensor_slices({'img_urls': img_urls, 'snd_urls': snd_urls})
        dataset = dataset.batch(batch_size=BATCH_SIZE)
        dataset = dataset.shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        print('Dataset created successfully!')
        print(f"Number of batches: {len(dataset)}")
        print(f"Number of images: {len(img_urls)}")
        print(f"Number of audios: {len(snd_urls)}")
        print(f"img_urls: {img_urls[:5]}")
        print(f"snd_urls: {snd_urls[:5]}")
        return dataset
    
if __name__ == '__main__':

    training_videos = [
        'data/test_video/test1.mp4',
        'data/test_video/test2.mp4',
        'data/test_video/test3.mp4'
    ]

    autoencoder = ConvolutionalAutoencoder()
    dataset = autoencoder.create_dataset(training_videos)
    print(dataset)

    # (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    # x_train = x_train.astype('float32') / 255.
    # x_test = x_test.astype('float32') / 255.
    # x_train = x_train[..., tf.newaxis]
    # x_test = x_test[..., tf.newaxis]

    # autoencoder = ConvolutionalAutoencoder()
    # autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    # autoencoder.fit(x_train, x_train,
    #                 epochs=1,
    #                 shuffle=True,
    #                 validation_data=(x_test, x_test))
    
    # # plot the models
    # tf.keras.utils.plot_model(autoencoder.encoder, to_file='convolutional_autoencoder_encoder.png', show_shapes=True, show_layer_names=True)
    # tf.keras.utils.plot_model(autoencoder.decoder, to_file='convolutional_autoencoder_decoder.png', show_shapes=True, show_layer_names=True)
    # tf.keras.utils.plot_model(autoencoder, to_file='convolutional_autoencoder.png', show_shapes=True, show_layer_names=True)

    # n = 10
    # x_test = x_test[:n]

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(20, 4))
    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(2, n, i + 1)
    #     plt.imshow(x_test[i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

    #     # display reconstruction
    #     ax = plt.subplot(2, n, i + 1 + n)
    #     plt.imshow(autoencoder(x_test)[i].numpy().reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

    # plt.savefig('output.png')
