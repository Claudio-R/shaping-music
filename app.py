import sys, os
import traceback
from src import FeaturesExtraction, DataEncoding, Generation, ClipGeneration

stages = [
    True,
    True,
    True,
    True
]

FPS = 15

if __name__ == "__main__":    

    if len(sys.argv) < 2: file_name = 'data/test_video/test2.mp4'
    else: file_name = sys.argv[1]

    if stages[0]:
        try:
            features = FeaturesExtraction.extract_features(file_name, FPS)
        except:
            print("\n### Error in extracting features from the video ###\n")
            traceback.print_exc()
        
    if stages[1]:
        try:
            i2sEncoder = DataEncoding.encode_data(features)
        except:
            print("\n### Error in encoding data ###\n")
            traceback.print_exc()
    
    if stages[2]:
        try:
            file_names = []
            for filename in os.listdir('data/mfe/audios'):
                path = os.path.join('data/mfe/audios', filename)
                file_names.append(path)
            file_names.sort()
            video_embeds_shapes, audio_embeds_shapes = i2sEncoder.get_shapes()
            gan = Generation.train_gan(file_names, video_embeds_shapes, audio_embeds_shapes)
            if stages[3]:  
                try:
                    ClipGeneration.create_videoclip(gan, 'data/test_video/test3.wav', FPS)
                except:
                    print("\n### Error in creating the clip ###\n")
                    traceback.print_exc()
        except:
            print("\n### Error in generating new images ###\n")
            traceback.print_exc()
