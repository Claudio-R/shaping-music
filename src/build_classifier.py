import tensorflow as tf
import yamnet
from multimodal_feature_extractor import FeatureExtractor

# IMAGES_FEATURES_EXTRACTOR_MODEL
image_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1', 
    'block4_conv1', 
    'block5_conv1'
    ]

image_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
image_model.trainable = False
outputs = [image_model.get_layer(name).output for name in image_layers]
image_FEM = tf.keras.Model(image_model.input, outputs)

# AUDIO MODEL
audio_layers = [
    'layer1/pointwise_conv',
    'layer2/pointwise_conv',
    'layer3/pointwise_conv',
    'layer4/pointwise_conv',
    'layer5/pointwise_conv',
    'layer6/pointwise_conv',
    'layer7/pointwise_conv',
    'layer8/pointwise_conv',
    'layer9/pointwise_conv',
    'layer10/pointwise_conv',
    'layer12/pointwise_conv',
    'layer13/pointwise_conv',
    'layer14/pointwise_conv',
    ]

audio_model = yamnet.inference.YamNet()
audio_model.trainable = False
outputs = [audio_model.get_layer(name).output for name in audio_layers]
audio_FEM = tf.keras.Model(audio_model.input, outputs)

# TEXT-TO-IMAGE GENERATOR
import yaml
import openai
with open("apikey.local.yml", "r") as stream:
    try:
        openai.api_key = yaml.safe_load(stream)['apikey']
    except yaml.YAMLError as exc:
        print(exc)
        print('Cannot load the APIKey')

classifier = FeatureExtractor(image_FEM, audio_FEM)
classifier('data/test_video/test1.mp4')