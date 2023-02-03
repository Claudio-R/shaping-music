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

    def __init__(self, image_model: tf.keras.Model, audio_model: tf.keras.Model):
        self.image_model = image_model
        self.audio_model = audio_model
        self.input_frames_path = 'data/test_frames'
        self.input_sounds_path = 'data/test_samples'
        self.full_audios_path = 'data/test_audio'
        self.generated_frames_path = 'data/generated_frames/'
        self.generated_clips_path = 'data/generated_clips/'

        input_shape = self.audio_model.layers[-1].output_shape
        output_shape = self.image_model.layers[-1].output_shape

        self.sound_to_frame_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape[1:]),
            tf.keras.layers.Conv2D(2048, 2, (2, 2), activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(output_shape[-1], activation='relu'),
        ])
        self.sound_to_frame_model.compile(optimizer='adam', loss='binary_crossentropy')

        print('image_model: ', self.image_model.summary())
        print('audio_model: ', self.audio_model.summary())
        print('sound_to_frame_model: ', self.sound_to_frame_model.summary())


    def __call__(self, video_url: str):
        assert video_url.endswith('.mp4'), 'FILE_FORMAT_ERROR_{}'.format(video_url)
        self.filename = video_url[0:-4]
        self.__preprocess_video(video_url)

        frm_list, snd_list = self.__retrieve_data()
        assert len(frm_list) == len(snd_list), 'DATA_PREPROCESSING_ERROR_FRAMES_{}_SOUNDS_{}'.format(len(frm_list), len(snd_list))
        
        audio_embed = []
        video_embed = []
        print('Encoding frames and sounds...')
        for i in range(len(frm_list)):
            img, snd = self.__encode(frm_list[i], snd_list[i])
            audio_embed.append(snd)
            video_embed.append(img)

        self.sound_to_frame_model.fit(audio_embed, video_embed, epochs=10, verbose=1)
        self.sound_to_frame_model.save('models/sound_to_frame_model.h5')
        self.__cleanup()

    def __getattribute__(self, __name: str) -> object:
        return super().__getattribute__(__name)

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

    @tf.function
    def __encode(self, _img: str, _snd: str):
        img, snd = self.__preprocess_data(_img, _snd)
        encoded_img = self.image_model(img)
        encoded_snd = self.audio_model(snd)
        return (encoded_img, encoded_snd)

    def __preprocess_data(self, _img: str, _snd: str):
        img = ImageUtils.load_img(_img)
        img = tf.keras.applications.vgg19.preprocess_input(img * 255)
        img = tf.image.resize(img, (224, 224))
        img = tf.Variable(img)

        wav_data, sr = sf.read(_snd, dtype=np.int16)
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        snd = wav_data / np.max(np.abs(wav_data))
        snd = snd.astype('float32')
        snd = tf.Variable(snd)

        if len(snd.shape) > 1:
            snd = np.mean(snd, axis=1)
        if sr != Params.sample_rate:
            snd = resampy.resample(snd, sr, Params.sample_rate)
        return (img, snd)

    def __cleanup(self):
        for filename in os.listdir(self.input_frames_path):
            os.remove(os.path.join(self.input_frames_path, filename))
        for filename in os.listdir(self.input_sounds_path):
            os.remove(os.path.join(self.input_sounds_path, filename))
        for filename in os.listdir(self.generated_frames_path):
            os.remove(os.path.join(self.generated_frames_path, filename))
        for filename in os.listdir(self.generated_clips_path):
            os.remove(os.path.join(self.generated_clips_path, filename))
        return