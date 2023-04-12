import tensorflow as tf
import tensorflow_hub as hub
import os, typing
import numpy as np
import imageio, librosa, soundfile as sf, moviepy.editor as mp
import tqdm

class ConvolutionalAutoencoder(tf.keras.Model):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        self.img_SIZE = 28
        self.style_extractor = self.define_style_extractor()
        self.snd_encoder = self.define_sound_encoder()
        self.encoder = self.define_encoder()
        self.decoder = self.define_decoder()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # plot the models
        tf.keras.utils.plot_model(self.style_extractor, to_file='style_extractor.png', show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(self.encoder, to_file='encoder.png', show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(self.decoder, to_file='decoder.png', show_shapes=True, show_layer_names=True)

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

    def call(self, img_urls:tf.Tensor, wav_urls:tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        '''
        img_urls: a tensor containing the images urls
        wav_urls: a tensor containing the wav urls
        '''
        imgs = tf.map_fn(self.preprocess_img, img_urls, dtype=tf.float32)
        embeds = self.style_extractor(imgs)
        styles = self.get_style(embeds)
        encoded_imgs = self.encoder(styles)
        decoded_imgs = self.decoder(encoded_imgs)

        # snd_embeds = tf.map_fn(self.compute_snd_embeds, wav_urls, dtype=tf.float32)
        snd_embeds = self.compute_snd_embeds(wav_urls)        
        return imgs, encoded_imgs, decoded_imgs, snd_embeds
    
    def custom_loss(self, imgs:tf.Tensor, encoded_imgs:tf.Tensor, decoded_imgs:tf.Tensor, snd_embeds:tf.Tensor):
        '''
        imgs: original images
        decoded_img: reconstructed image
        encoded_img: encoded image
        snd_embeds: sound embeddings
        '''
        # compute the reconstruction loss
        reconstruction_loss = tf.reduce_mean(tf.square(imgs - decoded_imgs))
        # compute the encoding loss
        encoding_loss = tf.reduce_mean(tf.square(encoded_imgs - snd_embeds))
        # compute the total loss
        total_loss = reconstruction_loss #+ encoding_loss
        return total_loss
        
    def train_step(self, data:tuple):
        img_urls, wav_urls = data['img_urls'], data['wav_urls']
        with tf.GradientTape() as tape:
            imgs, encoded_imgs, decoded_imgs, snd_embeds = self(img_urls, wav_urls)
            loss = self.custom_loss(imgs, encoded_imgs, decoded_imgs, snd_embeds)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def train(self, dataset:tf.data.Dataset, epochs:int=5):
        '''
        dataset: a tf.data.Dataset object containing the images urls and the corresponding wav urls files
        epochs: number of epochs to train the model
        '''
        for epoch in range(epochs):
            for data in tqdm.tqdm(dataset, desc=f"Epoch {epoch+1}", ):
                loss = self.train_step(data)
            print(f"loss: {loss}")

    def define_style_extractor(self) -> tf.keras.Model:
        vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        styles = ['block1_conv1', 'block2_conv1', 'block3_conv1']#, 'block4_conv1', 'block5_conv1']
        styles_layers = [vgg19.get_layer(style).output for style in styles]
        return tf.keras.Model(inputs=vgg19.input, outputs=styles_layers, name='style_extractor', trainable=False)

    def define_sound_encoder(self) -> tf.keras.Model:
        return hub.load('https://tfhub.dev/google/yamnet/1')

    @tf.autograph.experimental.do_not_convert
    def compute_snd_embeds(self, wav_urls:tf.Tensor) -> tf.Tensor:
        embeds = []
        for wav_url in wav_urls:
            wav_url = wav_url.numpy().decode('utf-8')
            wav, _ = librosa.load(wav_url, sr=16000, mono=True)
            wav = tf.convert_to_tensor(wav, dtype=tf.float32)
            _, e, _ = self.snd_encoder(wav)
            embeds.append(e)
        embeds = tf.stack(embeds)
        return embeds

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
        self.NUM_CHANNELS = 3
        input_layer = tf.keras.layers.Input(shape=(self.latent_dim), dtype=tf.float32, name='latent_input')
        dense_layer_1 = tf.keras.layers.Dense(self.latent_dim, activation='relu', name='decoder_dense_1')(input_layer)
        expand_layer = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, axis=1), axis=1))(dense_layer_1) # (None, 1, 1, 1024)
        conv2dTranspose_layer_1 = tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same', name='decoder_conv2d_transpose_1')(expand_layer) # (None, 8, 8, 8)
        conv2dTranspose_layer_2 = tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same', name='decoder_conv2d_transpose_2')(conv2dTranspose_layer_1) # (None, 16, 16, 16)
        dense_layer_2 = tf.keras.layers.Dense(self.img_SIZE * self.img_SIZE, activation='relu', name='decoder_dense_2')(conv2dTranspose_layer_2) # (None, 16, 16, 784)
        pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(dense_layer_2) # (None, 784)
        reshape_layer = tf.keras.layers.Reshape((self.img_SIZE, self.img_SIZE, 1))(pooling_layer) # (None, 28, 28, 1)
        dense_layer_3 = tf.keras.layers.Dense(self.NUM_CHANNELS, activation='relu', name='decoder_dense_3')(reshape_layer) # (None, 28, 28, 3)

        return tf.keras.Model(inputs=input_layer, outputs=dense_layer_3, name='decoder')
    
    def preprocess_img(self, img_url:tf.Tensor) -> tf.Tensor:
        img_url = img_url.numpy().decode('utf-8')
        img = tf.keras.preprocessing.image.load_img(img_url, target_size=(self.img_SIZE, self.img_SIZE))
        img = tf.keras.preprocessing.image.img_to_array(img, dtype=np.uint8)
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

    def parse_videos(self, video_urls:str, FPS:int=12, BATCH_SIZE:int=32) -> tf.data.Dataset:
        '''
        Parse videos into images and audios and return a tf.data.Dataset
        video_urls: list of video urls
        FPS: frames per second
        BATCH_SIZE: batch size
        '''

        self.fps = FPS
        self.batch_size = BATCH_SIZE

        img_urls, wav_urls = [], []
        img_idx, wav_idx = 0, 0

        # 0. Clear or create directories
        images_dir = 'data/autoencoder/images'
        audios_dir = 'data/autoencoder/sound'
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
            print(f"Original FPS: {original_fps}")
            new_fps = min(original_fps, FPS)
            print(f"New FPS: {new_fps}")
            sampling_period = 1 / new_fps
            t_starts = np.arange(0, video_duration, sampling_period)
            
            def augment_img(img, wav_path:str, img_idx:int, wav_idx:int):
                # resize the image
                img = tf.constant(img / 255.0, dtype=tf.float32) 
                img = tf.image.resize(img, (self.img_SIZE, self.img_SIZE)) 
                img = tf.cast(img * 255.0, tf.uint8) # (None, 28, 28, 3), tf.uint8

                # original image
                path = os.path.join(images_dir, f"frame_{img_idx}.jpg")
                imageio.imwrite(path, img)
                img_urls.append(path)
                wav_urls.append(wav_path)
                img_idx += 1
                wav_idx += 1

                # flipped image
                flipped_img = tf.image.flip_left_right(img)
                flipped_path = os.path.join(images_dir, f"frame_{img_idx}.jpg")
                imageio.imwrite(flipped_path, flipped_img)
                img_urls.append(flipped_path)
                wav_urls.append(wav_path)
                img_idx += 1
                wav_idx += 1
                
                # contrast image by a random value
                contrast_img = tf.image.adjust_contrast(img, np.random.uniform(0.1, 0.5))
                contrast_path = os.path.join(images_dir, f"frame_{img_idx}.jpg")
                imageio.imwrite(contrast_path, contrast_img)
                img_urls.append(contrast_path)
                wav_urls.append(wav_path)
                img_idx += 1
                wav_idx += 1
                
                # brightness image
                brightness_img = tf.image.adjust_brightness(img, np.random.uniform(0.1, 0.5))
                brightness_path = os.path.join(images_dir, f"frame_{img_idx}.jpg")
                imageio.imwrite(brightness_path, brightness_img)
                img_urls.append(brightness_path)
                wav_urls.append(wav_path)
                img_idx += 1
                wav_idx += 1
                
                # hue image
                hue_img = tf.image.adjust_hue(img, np.random.uniform(0.1, 0.5))
                hue_path = os.path.join(images_dir, f"frame_{img_idx}.jpg")
                imageio.imwrite(hue_path, hue_img)
                img_urls.append(hue_path)
                wav_urls.append(wav_path)
                img_idx += 1
                wav_idx += 1
                
                # saturation image
                saturation_img = tf.image.adjust_saturation(img, np.random.uniform(0.1, 0.5))
                saturation_path = os.path.join(images_dir, f"frame_{img_idx}.jpg")
                imageio.imwrite(saturation_path, saturation_img)
                img_urls.append(saturation_path)
                wav_urls.append(wav_path)
                img_idx += 1
                wav_idx += 1
                
                # gamma image
                gamma_img = tf.image.adjust_gamma(img, np.random.uniform(0.1, 0.5))
                gamma_path = os.path.join(images_dir, f"frame_{img_idx}.jpg")
                imageio.imwrite(gamma_path, gamma_img)
                img_urls.append(gamma_path)
                wav_urls.append(wav_path)
                img_idx += 1
                wav_idx += 1
                return img_idx, wav_idx
            
            print(f"Processing video {video_url} with {len(t_starts)-1} frames and {len(t_starts)-1} audio segments...")
            for t_start in tqdm.tqdm(t_starts[:-1]):
                # Use librosa to save audio segments
                wav, sr = librosa.load(music_path, sr=44100, offset=t_start, duration=sampling_period)
                wav_path = os.path.join(audios_dir, f"audio_{wav_idx}.wav")
                sf.write(wav_path, wav, sr)    
                # wav_urls.append(wav_path)     
                # wav_idx += 1

                # Use imageio to save frames
                img = video.get_frame(t_start)
                img_idx, wav_idx = augment_img(img, wav_path, img_idx, wav_idx)
                            
            video.close()
            music.close()

            print(f"Video {video_url} was successfully processed!")

        # 5. Create dataset
        assert len(img_urls) == len(wav_urls), "Number of images and audios must be equal!"
        dataset = tf.data.Dataset.from_tensor_slices({'img_urls': img_urls, 'wav_urls': wav_urls})
        dataset = dataset.batch(batch_size=BATCH_SIZE)
        dataset = dataset.shuffle(buffer_size=100).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        # 6. Print dataset info    
        print('\nDataset created successfully!')
        print(f"Number of batches: {len(dataset)}")
        print(f"Number of samples per batch: {BATCH_SIZE}")
        print(f"Total number of images and audio samples: {(len(img_urls), len(wav_urls))}\n")
   
        return dataset
    
if __name__ == '__main__':

    training_videos = [
        'data/test_video/test1.mp4',
        # 'data/test_video/test2.mp4',
        # 'data/test_video/test3.mp4'
    ]

    # load mnist dataset
    


    autoencoder = ConvolutionalAutoencoder()
    autoencoder.compile(optimizer='adam', loss=autoencoder.custom_loss)
    dataset = autoencoder.parse_videos(training_videos, FPS=60, BATCH_SIZE=32)
    autoencoder.train(dataset, epochs=1)

    n = 10
    dataset = dataset.take(n)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 4))
    for i, batch in enumerate(dataset):
        try: 
            # display original
            ax = plt.subplot(2, n, i + 1)
            img_url = batch['img_urls'][0]
            img = autoencoder.preprocess_img(img_url)

            # must be normalized to [0, 1] for imshow
            img = (img - img.min()) / (img.max() - img.min())

            plt.imshow(img)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # display reconstruction
            
            ax = plt.subplot(2, n, i + 1 + n)
            img = tf.expand_dims(img, axis=0)
            embeds = autoencoder.style_extractor(img)
            styles = autoencoder.get_style(embeds)
            encoded_img = autoencoder.encoder(styles)
            decoded_img = autoencoder.decoder(encoded_img)
            decoded_img = tf.squeeze(decoded_img, axis=0)

            # must be normalized to [0, 1] for imshow
            decoded_img = (decoded_img - tf.reduce_min(decoded_img)) / (tf.reduce_max(decoded_img) - tf.reduce_min(decoded_img))

            plt.imshow(decoded_img.numpy())
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        except Exception as e:
            print(e)
            continue

    plt.savefig('data/debug/convolutional_autoencoder.png')
    print('Done!')
 
