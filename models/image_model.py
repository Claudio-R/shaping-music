import tensorflow as tf
import numpy as np

SIZE = 56 # this should be 512 for the final model

class ImageModel(tf.keras.Sequential):
    '''
    VGG19-based model for embedding images
    '''
    def __init__(self, *args, **kwargs):
        super(ImageModel, self).__init__(*args, **kwargs)
        input_shape = (SIZE, SIZE, 3)
        input_layer = tf.keras.layers.InputLayer(input_shape)
        conv2D_layer = tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(4,4), padding='same', use_bias=False, name="adaptation_layer")
        vgg19 = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
        self.add(input_layer)
        self.add(conv2D_layer)
        for layer in vgg19.layers[1:-3]:
            self.add(layer)
        
    def __call__(self, img_url:str) -> tf.Tensor:
        img = self.__preprocess(img_url)
        return super(ImageModel, self).__call__(img)

    def __preprocess(self, img_url:str) -> tf.Tensor:
        img = self.__load_image(img_url)
        img = tf.keras.applications.vgg19.preprocess_input(img * 255)
        return tf.image.resize(img, (SIZE, SIZE))
    
    @staticmethod
    def __load_image(img_url:str) -> tf.Tensor:
        max_dim = SIZE
        img = tf.io.read_file(img_url)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img
    
    def predict(self, img: tf.Tensor) -> tf.Tensor:
        return super(ImageModel, self).__call__(img)

if __name__ == '__main__':
    img_model = ImageModel()
    img_model.compile(optimizer='adam', loss='categorical_crossentropy')
    img_model.summary()
    print('output shape: ', img_model.output_shape)