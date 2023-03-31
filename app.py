import os
import traceback
import argparse
from src import FeaturesExtraction, DataEncoding, Generation, ClipGeneration

if __name__ == "__main__":   

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='data/test_video/test3.mp4')
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--stages', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)

    args = parser.parse_args()
    FILE_NAME = args.file
    FPS = args.fps
    SIZE = args.size
    STAGES = args.stages
    EPOCHS = args.epochs

    if STAGES >= 1:
        try:
            features = FeaturesExtraction.extract_features(FILE_NAME, FPS, SIZE)
        except:
            print("\n### Error in extracting features from the video ###\n")
            traceback.print_exc()
        
    if STAGES >= 2:
        try:
            i2sEncoder = DataEncoding.encode_data(features, EPOCHS)
        except:
            print("\n### Error in encoding data ###\n")
            traceback.print_exc()
    
    if STAGES >= 3:
        try:
            file_names = []
            for filename in os.listdir('data/mfe/audios'):
                path = os.path.join('data/mfe/audios', filename)
                file_names.append(path)
            file_names.sort()
            video_embeds_shapes, audio_embeds_shapes = i2sEncoder.get_shapes()
            gan = Generation.train_gan(file_names, video_embeds_shapes, audio_embeds_shapes, SIZE, EPOCHS)
            if STAGES == 4:
                try:
                    ClipGeneration.create_videoclip(gan, 'data/test_video/test3.wav', FPS)
                except:
                    print("\n### Error in creating the clip ###\n")
                    traceback.print_exc()
        except:
            print("\n### Error in generating new images ###\n")
            traceback.print_exc()
