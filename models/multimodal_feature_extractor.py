from typing import Tuple
from utils.VideoUtils import preprocess_video
from utils.OutputUtils import print_progress_bar
from models.image_model import ImageModel
from models.sound_model import SoundModel
import tensorflow as tf

class MultimodalFeatureExtractor:
    '''
    A class that extracts features from a video and returns the embeddings for both image and audio content.
    It uses the ImageModel to extract the style from the frames and the SoundModel to extract the content from the audio.
    '''
    def __init__(self):       
        self.image_model = ImageModel()
        self.sound_model = SoundModel()

    def __call__(self, video_url: str) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        ''' Extracts features from a mp4 video and returns two lists containing the embeddings from images and frames.       '''
        frames, sounds, _ = preprocess_video(video_url)
        frame_embeds, sound_embeds = self.__compute_embeddings(frames, sounds)
        return frame_embeds, sound_embeds

    def __compute_embeddings(self, frame_urls: list, sound_urls: list) -> Tuple[list, list]:
        ''' 
        Computes embeddings for both image and audio content.
        returns: 
            - frames_embeds: list of lists containing the style tensors for each frame
            - sound_embeds: list of lists containing the style tensors for each sound
        '''
        frames_embeds, sound_embeds = [], []
        print('Feature extraction started.')
        for i in range(len(frame_urls)):
            frames_embeds.append(self.image_model(frame_urls[i]))
            sound_embeds.append(self.sound_model(sound_urls[i]))
            print_progress_bar(i+1, len(frame_urls), prefix="Computing embeddings:", length = 50, fill = '=')
        return frames_embeds, sound_embeds

    def predict_from_image(self, img: tf.Tensor) -> list:
        ''' Extracts features from a single image and returns the embeddings '''
        return self.image_model.predict(img)
    
    def predict_from_file(self, file_url:str) -> list:
        ''' Extracts features from a single image or audio and returns the embeddings '''
        file_type = file_url.split('.')[-1]
        if file_type == 'wav': return self.sound_model(file_url)
        elif file_type == 'jpg': return self.image_model(file_url)
        else: raise ValueError('Invalid file type. Must be either jpg or wav.')