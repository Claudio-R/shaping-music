import traceback
import numpy as np
from models.multimodal_feature_extractor import MultimodalFeatureExtractor

def extract_features(video_url:str):
    '''
    Extracts features from a mp4 video and returns two lists containing the embeddings from images and frames.
    '''
    print("Extracting features from the video: ", video_url)
    mfe = MultimodalFeatureExtractor()
    try: 
        video_embeds, audio_embeds = mfe(video_url)
        print("Video Embeddings Shape: ", video_embeds.shape)
        print("Audio Embeddings Shape: ", audio_embeds.shape)
        np.savez_compressed('data/embeddings/embeds.npz', video_embeds=video_embeds, audio_embeds=audio_embeds)
        return dict(video_embeds=video_embeds, audio_embeds=audio_embeds)
    except:
        print("Error in loading the feature extractor model")
        traceback.print_exc()