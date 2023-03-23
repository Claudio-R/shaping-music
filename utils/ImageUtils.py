import tensorflow as tf
import numpy as np
import PIL
import numpy as np

def upsample_image_url(img_url:str, size:int):
    img = tf.io.read_file(img_url)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (size, size), method='nearest')
    return img

if __name__ == "__main__":
    name = "training_image_at_epoch_0003"
    image = upsample_image_url(f"data/debug/{name}.jpg", 512)
    PIL.Image.fromarray((image * 255).numpy().astype(np.uint8)).save(f"data/debug/{name}_up.jpg")
