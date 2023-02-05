import tensorflow as tf

class ImageModel(tf.keras.Model):
    '''
    VGG19-based model for embedding images
    '''
    def __init__(self):
        output_layers = ['block4_pool']
        vgg19 = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
        output_layers = ['flatten']
        output_layers = [vgg19.get_layer(name).output for name in output_layers]
        super(ImageModel, self).__init__(inputs=vgg19.input, outputs=output_layers, name='ImageEmbeddingsExtractor')

    def __call__(self, img_url:str) -> tf.Tensor:
        '''
        Returns the embedding of the image at img_url
        '''
        img = self.__preprocess(img_url)
        return super(ImageModel, self).__call__(img)

    def __preprocess(self, img_url:str) -> tf.Tensor:
        img = self.__load_image(img_url)
        img = tf.keras.applications.vgg19.preprocess_input(img * 255)
        return tf.image.resize(img, (224, 224))
    
    @staticmethod
    def __load_image(img_url:str) -> tf.Tensor:
        max_dim = 512
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

    def get_dataset(self, img_urls:list) -> tf.data.Dataset:
        '''
        Returns a tf.data.Dataset containing the embeddings of the images in img_urls
        '''
        images = [self.__preprocess(url) for url in img_urls]
        return tf.data.Dataset.from_tensor_slices(images).map(lambda x: self(x))

if __name__ == '__main__':
    img_model = ImageModel()
    img_model.summary()
    print(img_model('data/test_frames/frame1.jpg'))