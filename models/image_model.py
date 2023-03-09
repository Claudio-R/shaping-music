from typing import List
import tensorflow as tf
from utils.OutputUtils import print_progress_bar

# SIZE = 64 # this should be 512 for the final model

class ImageModel(tf.Module):
    '''
    VGG19-based model for style extraction from images.
    '''
    def __init__(self, SIZE=64):
        self.SIZE = tf.constant(SIZE, dtype=tf.int32)
        vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        # outputs = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        outputs = ['block1_conv1', 'block2_conv1']
        if type(outputs) != list: outputs = [outputs]
        self.model = tf.keras.Model(
            inputs=vgg19.input, 
            outputs=[vgg19.get_layer(output).output for output in outputs], 
            name='image_model'
            )
        
    # @tf.function
    def __call__(self, img_urls:list) -> List[List[tf.Tensor]]:
        ''' Extracts the style from a list of images '''
        embeddings = []
        for i, img_url in enumerate(img_urls):
            img = self.__preprocess(img_url)
            outputs = self.model(img)
            if type(outputs) != list: outputs = [outputs]
            style_outputs = [self.extract_style(output) for output in outputs]
            embeddings.append(style_outputs)
            print_progress_bar(i+1, len(img_urls), prefix="Extracting video features:", length = 50, fill = '=')
        return embeddings

    def __preprocess(self, img_url:str) -> tf.Tensor:
        # 1. load the image
        img = self.load_image(img_url)
        # 2. resize the image
        min_shape = tf.reduce_min(tf.shape(img)[:-1])
        img = tf.image.resize_with_crop_or_pad(img, min_shape, min_shape)
        img = tf.image.resize(img, (self.SIZE, self.SIZE))
        # 3. convert to BGR and scale to 0-255
        img = tf.keras.applications.vgg19.preprocess_input(img[tf.newaxis, :] * 255)
        # 4. save the image for debugging
        tf.keras.preprocessing.image.save_img('data/debug/test.jpg', img[0])
        return img
    
    @staticmethod
    def load_image(img_url:str) -> tf.Tensor:
        img = tf.io.read_file(img_url)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    @staticmethod
    def extract_style(input_tensor: tf.Tensor) -> tf.Tensor:
        ''' Extracts the style from a tensor computing the Gram matrix '''
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)
    
    def predict(self, img: tf.Tensor) -> tf.Tensor:
        ''' Predicts the style from an image '''
        # 1. resize the image
        img = tf.image.convert_image_dtype(img, tf.float32)
        min_shape = tf.reduce_min(tf.shape(img)[:-1])
        img = tf.image.resize_with_crop_or_pad(img, min_shape, min_shape)
        img = tf.image.resize(img, (self.SIZE, self.SIZE))
        # 2. convert to BGR and scale to 0-255
        img = tf.keras.applications.vgg19.preprocess_input(img[tf.newaxis, :] * 255)
        # 3. extract the style
        outputs = self.model(img)
        style_outputs = [self.extract_style(output) for output in outputs]
        return style_outputs
    
    def summary(self):
        return self.model.summary()

    def plot(self):
        tf.keras.utils.plot_model(self.model, to_file='data/debug/image_model.png', show_shapes=True, show_layer_names=True)

if __name__ == '__main__':
    img_model = ImageModel()
    img_model.model.compile(optimizer='adam', loss='categorical_crossentropy')
    print('input shape: ', img_model.model.input_shape)
    print('output shape: ', img_model.model.output_shape)
    img_model.model.summary()
    tf.keras.utils.plot_model(img_model.model, to_file='data/debug/image_model.png', show_shapes=True, show_layer_names=True)
    