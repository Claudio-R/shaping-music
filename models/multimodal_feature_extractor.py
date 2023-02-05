import os, sys
from typing import Tuple
from utils.VideoUtils import preprocess_video
from multipledispatch import dispatch
from models.image_model import ImageModel
from models.sound_model import SoundModel
import tensorflow as tf

class MultimodalFeatureExtractor:
    '''
    Class responsible for multimodal encoding.
    Given a video url, produces two lists containing embeddings for both image and audio content.
    sound_model: pretrained tf.Model for audio classification
    image_model: pretrained tf.Model for image classification
    '''
    def __init__(self):
        self.input_frames_path = 'data/test_frames'
        self.input_sounds_path = 'data/test_samples'        
        self.image_model = ImageModel()
        self.sound_model = SoundModel()
        print('MultimodalFeatureExtractor initialized.')

    
    # TODO: declare this call a tf.function fixing bug on load_image() function
    @dispatch(str)
    def __call__(self, video_url: str) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        '''
        Extracts features from a mp4 video and returns two lists containing the embeddings from images and frames.
        '''
        imgs_list, wavs_list = self.__process_video(video_url)     

        audio_embeds = []
        video_embeds = []
        for i in range(len(imgs_list)):
            img, snd = self.__compute_embeddings(imgs_list[i], wavs_list[i])
            audio_embeds.append(snd)
            video_embeds.append(img)
            # add progress bar
            sys.stdout.write('\r')
            sys.stdout.write('Progress: %d/%d' % (i+1, len(imgs_list)))
            sys.stdout.flush()
        sys.stdout.write('\n')

        
        # convert to stacks 
        audio_embeds = tf.stack(audio_embeds)
        video_embeds = tf.stack(video_embeds)

        return video_embeds, audio_embeds
    
    @dispatch(str, str)
    def __call__(self, file_type:str, file_url:str) -> list:
        '''
        Extracts features from an audio or video and returns two lists containing the embeddings.
        '''
        if file_type == 'image':
            embeds = self.image_model(file_url)
        elif file_type == 'sound':
            embeds = self.sound_model(file_url)
        else:
            raise ValueError('Invalid file type. Must be either image or sound.')
        return embeds

    def __del__(self):
        self.__cleanup()

    def __process_video(self, video_url: str):
        preprocess_video(video_url, self.input_frames_path, self.input_sounds_path)
        frames_urls = []
        sounds_urls = []
        for filename in os.listdir(self.input_frames_path):
            path = os.path.join(self.input_frames_path, filename)
            frames_urls.append(path)
        frames_urls.sort()

        for filename in os.listdir(self.input_sounds_path):
            path = os.path.join(self.input_sounds_path, filename)
            sounds_urls.append(path)
        sounds_urls.sort()
        
        # this should be fixed in the preprocessing step
        if (len(frames_urls) > len(sounds_urls)):
            while(len(frames_urls) != len(sounds_urls)):
                frames_urls.pop()
        elif (len(frames_urls) < len(sounds_urls)):
            while(len(frames_urls) != len(sounds_urls)):
                sounds_urls.pop()
        return (frames_urls, sounds_urls)

    def __compute_embeddings(self, img_url: str, wav_url: str):
        img_embed = self.image_model(img_url)
        wav_embed = self.sound_model(wav_url)
        return (img_embed, wav_embed)
    
    # def get_output_shapes(self):
    #     '''
    #     Returns the output shapes of the image and sound models
    #     '''
    #     return (self.image_model.output_shape, self.sound_model.output_shape)

    def get_dataset(self, files_type:str, urls: list):
        '''
        Returns a tf.Dataset containing the embeddings for the given urls
        '''
        if files_type == 'image':
            dataset = self.image_model.get_dataset(urls)
        elif files_type == 'sound':
            dataset = self.sound_model.get_dataset(urls)
        else:
            raise ValueError('Invalid file type. Must be either image or sound.')
        return dataset


    def __cleanup(self):
        '''
        Cleans up the input folders
        '''
        print("\nCleaning up...")
        # for filename in os.listdir(self.input_frames_path):
        #     path = os.path.join(self.input_frames_path, filename)
        #     os.remove(path)
        # for filename in os.listdir(self.input_sounds_path):
        #     path = os.path.join(self.input_sounds_path, filename)
        #     os.remove(path)
    