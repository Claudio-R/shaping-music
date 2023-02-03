import traceback
import numpy as np
from models.multimodal_feature_extractor import MultimodalFeatureExtractor

def build_feature_extractor():
    print("Defining the Feature Extractor")
    feature_extractor = MultimodalFeatureExtractor()
    return feature_extractor

def extract_features(video_url:str):
    '''
    Extracts features from a mp4 video and returns two lists containing the embeddings from images and frames.
    '''
    print("Extracting features from the video: ", video_url)
    feature_extractor = build_feature_extractor()
    try: 
        video_embeds, audio_embeds = feature_extractor(video_url)
        video_shape, audio_shape = feature_extractor.get_output_shapes()
        np.savez_compressed('data/processed/embeddings.npz', video_embeds=video_embeds, audio_embeds=audio_embeds, video_shape=video_shape, audio_shape=audio_shape)
        return dict(video_embeds=video_embeds, audio_embeds=audio_embeds, video_shape=video_shape, audio_shape=audio_shape)
    except:
        print("Error in loading the feature extractor model")
        traceback.print_exc()