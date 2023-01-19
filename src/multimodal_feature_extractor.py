import os
import tensorflow as tf
import PIL
import requests
from io import BytesIO
from utils import load_image, videotest
import moviepy
from moviepy.editor import VideoFileClip, AudioFileClip
import openai

class FeatureExtractor:
    '''
    Class responsible for multimodal classification.
    Given an audio track and the corresponding sequence of frames to classify, produces a string by concatenating labels.
    sound_model: pretrained tf.Model for audio classification
    image_model: pretrained tf.Model for image classification
    '''

    def __init__(self, image_model, audio_model):
        self.image_model = image_model
        self.audio_model = audio_model
        self.frame_count = 0
        self.fps = 2
        self.frames_path = 'data/test_frames'
        self.sounds_path = 'data/test_samples/'
        self.full_audios_path = 'data/test_audio'
        self.generated_frames_path = 'data/generated_frames/'
        self.generated_clips_path = 'data/generated_clips/'


    def __call__(self, video_url):
        
        if(video_url.endswith('.mp4') == False):
            return
        self.filename = video_url[0:-4]

        self.frame_count = 0
        self.__preprocess_video(video_url)

        img_list, snd_list = self.__retrieve_data()
        
        for i in range(len(img_list)):
            self.__classify(img_list[i], snd_list[i])
        print('Completed! Generating clip...')
        
        self.__generate_clip()
        
        self.__cleanup()
        
        print('DONE!')


    def __preprocess_video(self, video_url):
        videotest.main(video_url)
    
    
    def __retrieve_data(self):
        frames_list = []
        for filename in os.listdir(self.frames_path):
            path = os.path.join(self.frames_path, filename)
            frames_list.append(path)

        sounds_list = []
        for filename in os.listdir(self.sounds_path):
            path = os.path.join(self.sounds_path, filename)
            frames_list.append(path)
            
        return (frames_list, sounds_list)


    def __classify(self, img, snd):

        # Image Classification
        img = load_image(img)
        input_img = tf.keras.applications.vgg19.preprocess_input(img * 255)
        input_img = tf.image.resize(input_img, (224, 224))
        predictions_img = self.image_model(input_img)
        predictions_img = tf.keras.applications.vgg19.decode_predictions(predictions_img.numpy())[0]

        # Sound Classification
        predictions_snd = self.audio_model(snd)

        # Clip Generation
        text = self.__parse_predictions(predictions_img, predictions_snd)
        try:
            self.__generate_image(text)
        except:
            try: 
                print(text)
                print('Your prompt may contain text that is not allowed by our safety system.')
                text = predictions_snd
                self.__generate_image(text)
            except:
                try:
                    print(text)
                    print('Your prompt may contain text that is not allowed by our safety system.')
                    text = self.__parse_predictions(predictions_img, [])
                    self.__generate_image(text)
                except:
                    print(text)
                    print('Your prompt may contain text that is not allowed by our safety system.')
                    self.__generate_image(predictions_snd[0])


    def __parse_predictions(self, predictions_img, predictions_snd):
        predictions_img = [class_name for (number, class_name, prob) in predictions_img]
        predictions_img = [class_name.split('_') for class_name in predictions_img]
        input_text = predictions_snd
        for words in predictions_img:
            for w in words:
                input_text.append(w)
        return ' '.join(input_text)


    def __generate_image(self, text):
        response = openai.Image.create(
            prompt = text,
            n=1,
            size="512x512"
        )
        image_url = response['data'][0]['url']
        response = requests.get(image_url)
        img = PIL.Image.open(BytesIO(response.content))
        self.__store_image(img)


    def __store_image(self, img):
        generated_frame_path = 'img_at_frame_{}.png'.format(self.frame_count)
        path = os.path.join(self.generated_frames_path, generated_frame_path)
        tf.keras.utils.save_img(path, img)
        self.frame_count += 1


    def __generate_clip(self):
        output_file_path = os.path.join(self.generated_clips_path, 'Test.mp4')
        
        generated_frames_list = []
        for filename in os.listdir(self.images_path):
            path = os.path.join(self.images_path, filename)
            generated_frames_list.append(path)

        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(generated_frames_list, self.fps)
        clip.write_videofile(output_file_path)

        audio_clip = AudioFileClip(self.audio_path)
        video_clip = VideoFileClip(output_file_path).set_audio(audio_clip)
        video_clip.write_videofile(output_file_path)
    

    def __cleanup(self):
        return