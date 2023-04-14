# NOTE: mnist images are normalized in the range [0, 1]
# TODO: define a custom loss function

import tensorflow as tf
import tqdm, os
import imageio, moviepy.editor as mp
import librosa, soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

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
            print(f'\nEpoch {e+1}/{epochs}')
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
            
    def train_step(self, batch:tf.Tensor):
        with tf.GradientTape() as tape:
            decoded_img = self.call(batch)
            loss = self.custom_loss(batch, decoded_img)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
    
    def test_step(self, batch:tf.Tensor):
        decoded_img = self.call(batch)
        loss = self.custom_loss(batch, decoded_img)
        return loss

    def custom_loss(self, y_true:tf.Tensor, y_pred:tf.Tensor):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y_true, y_pred)

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
        normalize_layer = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(dense_layer_2)
        return tf.keras.Model(inputs=input_layers, outputs=normalize_layer, name='encoder')

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
        dataset = tf.data.Dataset.from_tensor_slices({'img_urls': img_urls, 'wav_urls': wav_urls}).shuffle(len(img_urls))
        train_data, test_data = dataset.take(int(len(dataset) * 0.8)), dataset.skip(int(len(dataset) * 0.8))
        train_data, val_data = train_data.take(int(len(train_data) * 0.8)), train_data.skip(int(len(train_data) * 0.8))
        dataset = {
            'train': train_data.batch(BATCH_SIZE),
            'val': val_data.batch(BATCH_SIZE),
            'test': test_data.batch(BATCH_SIZE)
        }

        # 6. Print dataset info    
        print('\nDataset created successfully!\n')
        print(f"Sample: {dataset['train'].take(1).element_spec}")
        print(f"Number of training samples: {len(train_data) * BATCH_SIZE}")
        print(f"Number of validation samples: {len(val_data) * BATCH_SIZE}")
        print(f"Number of testing samples: {len(test_data) * BATCH_SIZE}")
        print(f"Total number of samples: {len(img_urls)}")
        print(f"Batch size: {BATCH_SIZE}\n")

        return dataset
    
if __name__ == '__main__':

    autoencoder = ConvolutionalAutoencoder()

    training_videos = [
        'data/test_video/test1.mp4',
        # 'data/test_video/test2.mp4',
        # 'data/test_video/test3.mp4'
    ]

    # load mnist dataset
    
    dataset = autoencoder.parse_videos(training_videos, FPS=60, BATCH_SIZE=32)

    # autoencoder.train(dataset, epochs=1)
    
    # n = 10
    # dataset = dataset.take(n)

    # plt.figure(figsize=(20, 4))
    # for i, batch in enumerate(dataset):
    #     try: 
    #         # display original
    #         ax = plt.subplot(2, n, i + 1)
    #         img_url = batch['img_urls'][0]
    #         img = autoencoder.preprocess_img(img_url)

    #         # must be normalized to [0, 1] for imshow
    #         img = (img - img.min()) / (img.max() - img.min())

    #         plt.imshow(img)
    #         plt.gray()
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)
    #         # display reconstruction
            
    #         ax = plt.subplot(2, n, i + 1 + n)
    #         img = tf.expand_dims(img, axis=0)
    #         embeds = autoencoder.style_extractor(img)
    #         styles = autoencoder.get_style(embeds)
    #         encoded_img = autoencoder.encoder(styles)
    #         decoded_img = autoencoder.decoder(encoded_img)
    #         decoded_img = tf.squeeze(decoded_img, axis=0)

    #         # must be normalized to [0, 1] for imshow
    #         decoded_img = (decoded_img - tf.reduce_min(decoded_img)) / (tf.reduce_max(decoded_img) - tf.reduce_min(decoded_img))

    #         plt.imshow(decoded_img.numpy())
    #         plt.gray()
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)

    #     except Exception as e:
    #         print(e)
    #         continue

    # plt.savefig('data/debug/convolutional_autoencoder.png')
    print('Done!')
 


# if __name__ == '__main__':
#     (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
#     x_train = x_train.astype('float32') / 255.
#     x_test = x_test.astype('float32') / 255.
#     x_train = x_train[..., tf.newaxis]
#     x_test = x_test[..., tf.newaxis]

#     dataset_size = len(x_train)
#     x_train = tf.data.Dataset.from_tensor_slices(x_train).shuffle(dataset_size)
#     x_train, x_val = x_train.take(int(dataset_size * 0.8)), x_train.skip(int(dataset_size * 0.8))
#     x_train = x_train.batch(32).shuffle(1000)
#     x_val = x_val.batch(32).shuffle(1000)
#     x_test = tf.data.Dataset.from_tensor_slices(x_test).batch(32).shuffle(1000)

#     autoencoder = ConvolutionalAutoencoder()
#     autoencoder.train(x_train, x_val, epochs=3)
        
#     n = 10
#     x_test = x_test.take(n).as_numpy_iterator().next()

#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(20, 4))
#     for i in range(n):
#         # display original
#         ax = plt.subplot(2, n, i + 1)
#         img = x_test[i].reshape(28, 28)
#         plt.imshow(img)
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#         # display reconstruction
#         ax = plt.subplot(2, n, i + 1 + n)

#         img = autoencoder(tf.expand_dims(x_test[i], axis=0)).numpy().reshape(28, 28)
#         plt.imshow(img)
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
   
#     plt.savefig('data/debug/convolutional_autoencoder.png')
#     print('Done!')
 