import os
import tensorflow as tf
import soundfile as sf
import resampy
import PIL
import requests
from io import BytesIO
from utils import ImageUtils, VideoUtils
import moviepy
from moviepy.editor import VideoFileClip, AudioFileClip
import openai
import numpy as np

from yamnet.params import Params

class FeatureExtractor:
    '''
    Class responsible for multimodal encoding.
    Given an audio track and the corresponding sequence of frames to classify, produces a string by concatenating labels.
    sound_model: pretrained tf.Model for audio classification
    image_model: pretrained tf.Model for image classification
    '''

    def __init__(self, image_model, audio_model):
        self.image_model = image_model
        self.audio_model = audio_model
        self.frame_count = 0
        self.fps = 2
        self.input_frames_path = 'data/test_frames'
        self.input_sounds_path = 'data/test_samples'
        self.full_audios_path = 'data/test_audio'
        self.generated_frames_path = 'data/generated_frames/'
        self.generated_clips_path = 'data/generated_clips/'

        #define a model able to map a sound to a frame
        input_shape = self.sound_model.layers[-1].output_shape
        output_shape = self.image_model.layers[-1].output_shape
        self.sound_to_frame_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(units=(3, 3, 64), activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
            tf.keras.layers.Dense(output_shape[1], activation='relu'),
            tf.keras.layers.Reshape(output_shape[1:])
        ])

    def __call__(self, video_url):
        assert video_url.endswith('.mp4'), 'FILE_FORMAT_ERROR_{}'.format(video_url)
        self.filename = video_url[0:-4]
        self.frame_count = 0
        self.__preprocess_video(video_url)

        frm_list, snd_list = self.__retrieve_data()
        assert len(frm_list) == len(snd_list), 'DATA_PREPROCESSING_ERROR_FRAMES_{}_SOUNDS_{}'.format(len(frm_list), len(snd_list))

        for i in range(len(frm_list)):
            self.__autoencode(frm_list[i], snd_list[i])
        print('Completed! Generating clip...')
        # self.__generate_clip()
        self.__cleanup()
        
        print('DONE!')

    def __preprocess_video(self, video_url):
        VideoUtils.main(video_url)
    
    def __retrieve_data(self):
        frames_list = []
        for filename in os.listdir(self.input_frames_path):
            path = os.path.join(self.input_frames_path, filename)
            frames_list.append(path)
        frames_list.sort()

        sounds_list = []
        for filename in os.listdir(self.input_sounds_path):
            path = os.path.join(self.input_sounds_path, filename)
            sounds_list.append(path)
        sounds_list.sort()
        
        # ensures the lists have same length
        if (len(frames_list) > len(sounds_list)):
            while(len(frames_list) != len(sounds_list)):
                frames_list.pop()
        return (frames_list, sounds_list)

    def __autoencode(self, _img, _snd):
        img = ImageUtils.load_img(_img)
        img = tf.keras.applications.vgg19.preprocess_input(img * 255)
        img = tf.image.resize(img, (224, 224))
        encoded_img = self.image_model(img)
        
        wav_data, sr = sf.read(_snd, dtype=np.int16)
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        snd = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        snd = snd.astype('float32')

        # Convert to mono and the sample rate expected by YAMNet.
        if len(snd.shape) > 1:
            snd = np.mean(snd, axis=1)
        if sr != Params.sample_rate:
            snd = resampy.resample(snd, sr, Params.sample_rate)

        encoded_snd = self.audio_model(snd)

        return (encoded_img, encoded_snd)

    def __classify(self, img, snd):

        # Image Classification
        img = ImageUtils.load_img(img)
        input_img = tf.keras.applications.vgg19.preprocess_input(img * 255)
        input_img = tf.image.resize(input_img, (224, 224))
        predictions_img = self.image_model(input_img)
        predictions_img = tf.keras.applications.vgg19.decode_predictions(np.ndarray(predictions_img))[0]

        # Sound Classification
        predictions_snd = self.audio_model(snd)

        # Clip Generation
        text = self.__parse_predictions(predictions_img, predictions_snd)
        try:
            self.__generate_image(text)
        except:
            try: 
                print(text)
                print('Your prompt may contain text that is not allowed by our safety system.')
                text = predictions_snd
                self.__generate_image(text)
            except:
                try:
                    print(text)
                    print('Your prompt may contain text that is not allowed by our safety system.')
                    text = self.__parse_predictions(predictions_img, [])
                    self.__generate_image(text)
                except:
                    print(text)
                    print('Your prompt may contain text that is not allowed by our safety system.')
                    self.__generate_image(predictions_snd[0])

    def __parse_predictions(self, predictions_img, predictions_snd):
        predictions_img = [class_name for (number, class_name, prob) in predictions_img]
        predictions_img = [class_name.split('_') for class_name in predictions_img]
        input_text = predictions_snd
        for words in predictions_img:
            for w in words:
                input_text.append(w)
        return ' '.join(input_text)

    def __generate_image(self, text):
        response = openai.Image.create(
            prompt = text,
            n=1,
            size="512x512"
        )
        image_url = response['data'][0]['url']
        response = requests.get(image_url)
        img = PIL.Image.open(BytesIO(response.content))
        self.__store_image(img)

    def __store_image(self, img):
        generated_frame_path = 'img_at_frame_{}.png'.format(self.frame_count)
        path = os.path.join(self.generated_frames_path, generated_frame_path)
        tf.keras.utils.save_img(path, img)
        self.frame_count += 1

    def __generate_clip(self):
        output_file_path = os.path.join(self.generated_clips_path, 'Test.mp4')
        
        generated_frames_list = []
        for filename in os.listdir(self.generated_frames_path):
            path = os.path.join(self.generated_frames_path, filename)
            generated_frames_list.append(path)

        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(generated_frames_list, self.fps)
        clip.write_videofile(output_file_path)

        audio_clip = AudioFileClip(self.audio_path)
        video_clip = VideoFileClip(output_file_path).set_audio(audio_clip)
        video_clip.write_videofile(output_file_path)
    
    def __cleanup(self):
        return