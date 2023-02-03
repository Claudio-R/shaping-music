import tensorflow as tf
from utils.ImageUtils import load_img

class ImageModel(tf.keras.Model):
    '''
    VGG19-based model for embedding images
    '''
    def __init__(self):
        output_layers = ['block4_pool']
        vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg19.trainable = False
        output_layers = [vgg19.get_layer(name).output for name in output_layers]
        pooling_layer = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=None)(output_layers[-1])
        super(ImageModel, self).__init__(inputs=vgg19.input, outputs=pooling_layer, name='ImageEmbeddingsExtractor')

    def __call__(self, img_url:str) -> tf.Tensor:
        '''
        Returns the embedding of the image at img_url
        '''
        img = self.__preprocess(img_url)
        return super(ImageModel, self).__call__(img)

    def __preprocess(self, img_url:str) -> tf.Tensor:
        img = load_img(img_url)
        img = tf.keras.applications.vgg19.preprocess_input(img * 255)
        return tf.image.resize(img, (224, 224))
    
    def get_dataset(self, img_urls:list) -> tf.data.Dataset:
        '''
        Returns a tf.data.Dataset containing the embeddings of the images in img_urls
        '''
        images = [self.__preprocess(url) for url in img_urls]
        return tf.data.Dataset.from_tensor_slices(images).map(lambda x: self(x))

if __name__ == '__main__':
    img_model = ImageModel()
    img_model.summary()
    img_model('data/test_frames/frame1.jpg')