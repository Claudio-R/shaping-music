import sys, os
import traceback
from src import FeaturesExtraction, DataEncoding, Generation, ClipGeneration

stages = [
    True,
    False,
    False,
    False
]

if __name__ == "__main__":    

    if len(sys.argv) < 2: file_name = 'data/test_video/test2.mp4'
    else: file_name = sys.argv[1]

    if stages[0]:
        try:
            features = FeaturesExtraction.extract_features(file_name)
        except:
            print("Error in extracting features from the video")
            traceback.print_exc()
        
    if stages[1]:
        try:
            encoder = DataEncoding.encode_data('data/embeddings/embeds.npz')
        except:
            print("Error in encoding data")
            traceback.print_exc()
    
    if stages[2]:
        try:
            file_names = []
            for filename in os.listdir('data/test_samples'):
                path = os.path.join('data/test_samples', filename)
                file_names.append(path)
            file_names.sort()
            gan = Generation.train_gan(file_names)
        except:
            print("Error in generating new images")
            traceback.print_exc()

    if stages[3]:
        try:
            # ClipGeneration.create_clip_from_url(file_name, 'data/test_frames', 'data/test_samples')
            ClipGeneration.create_videoclip('data/test_audio/test.wav', 'data/test_samples')
        except:
            print("Error in creating the clip")
            traceback.print_exc()
