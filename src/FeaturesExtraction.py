import traceback
from typing import Dict
from models.multimodal_feature_extractor import MultimodalFeatureExtractor

def extract_features(video_url:str, fps:int=2) -> Dict[str, list]:
    '''
    Extracts features from a mp4 video and returns two lists containing the embeddings from images and frames.
    '''
    mfe = MultimodalFeatureExtractor()
    try: 
        embeds = mfe(video_url, fps)
        video_embeds = embeds['video_embeds']
        audio_embeds = embeds['audio_embeds']

        print("\nVIDEO EMBEDDINGS:")
        print(" - Number of frames: ", len(video_embeds))
        for i in range(len(video_embeds[0])):
            print(" - Embedding {} Shape: {}".format(i, video_embeds[0][i].shape))
        
        print("\nAUDIO EMBEDDINGS:")
        print(" - Number of frames: ", len(audio_embeds))
        for i in range(len(audio_embeds[0])):
            print(" - Embedding {} Shape: {}".format(i, audio_embeds[0][i].shape))
        
        return embeds
    
    except:
        print("Error in loading the feature extractor model")
        traceback.print_exc()