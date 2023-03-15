from typing import List, Dict
from utils.VideoUtils import preprocess_video
from models.image_model import ImageModel
from models.sound_model import SoundModel
import tensorflow as tf

class MultimodalFeatureExtractor(tf.Module):
    '''
    A class that extracts features from a video and returns the embeddings for both image and audio content.
    It uses the ImageModel to extract the style from the frames and the SoundModel to extract the content from the audio.
    '''
    def __init__(self):       
        self.image_model = ImageModel()
        self.sound_model = SoundModel()

    def __call__(self, video_url: str) -> Dict[str, list]:
        ''' Extracts features from a mp4 video and returns two lists containing the embeddings from images and frames.       '''
        print('\nExtracting features from video:')
        frame_urls, audio_urls, _ = preprocess_video(video_url)
        return self.__compute_embeddings(frame_urls, audio_urls)

    def __compute_embeddings(self, frame_urls: list, audio_urls: list) -> Dict[str, list]:
        ''' 
        Computes embeddings for both image and audio content.
        returns: 
            - video_embeds: list of lists containing the style tensors for each frame
            - audio_embeds: list containing the content tensors for each audio segment
        '''
        return {'video_embeds': self.image_model(frame_urls), 'audio_embeds': self.sound_model(audio_urls)}

    def predict_from_image(self, img: tf.Tensor) -> List[tf.Tensor]:
        ''' Extracts features from a single image and returns the embeddings '''
        return self.image_model.predict(img)
    
    def predict_from_file(self, file_url:str, verbose:bool=True) -> List[tf.Tensor]:
        ''' Extracts features from a single image or audio and returns the embeddings '''
        file_type = file_url.split('.')[-1]
        if file_type == 'wav': return self.sound_model([file_url], verbose=verbose)
        elif file_type == 'jpg': return self.image_model([file_url])
        else: raise ValueError('Invalid file type. Must be either jpg or wav.')