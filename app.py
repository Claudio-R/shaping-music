import sys, os
import traceback
from src import FeaturesExtraction, DataEncoding, Generation
if __name__ == "__main__":    

    '''
    This is the main file of the project. It is responsible for calling the other modules.
    The first argument is the video file name.
    The second argument is a boolean value that indicates whether the features should be extracted.
    The third argument is a boolean value that indicates whether the data should be encoded.
    The fourth argument is a boolean value that indicates whether the new images should be generated.
    '''

    if len(sys.argv) < 2: file_name = 'data/test_video/test1.mp4'
    else: file_name = sys.argv[1]

    if sys.argv[2]:
        try:
            features = FeaturesExtraction.extract_features(file_name)
        except:
            print("Error in extracting features from the video")
            traceback.print_exc()
        
    if sys.argv[3]:
        try:
            encoder = DataEncoding.encode_data('data/processed/embeddings.npz')
        except:
            print("Error in encoding data")
            traceback.print_exc()
    
    if sys.argv[4]:
        try:
            file_names = []
            for filename in os.listdir('data/test_samples'):
                file_names.append(filename)
            file_names.sort()
            gan = Generation.train_gan(file_names)

        except:
            print("Error in generating new images")
            traceback.print_exc()
