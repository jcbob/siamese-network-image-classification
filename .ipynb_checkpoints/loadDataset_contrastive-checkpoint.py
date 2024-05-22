import tensorflow as tf
import numpy as np
import random
import os

class MapFunction():
	def __init__(self, imageSize):
		# define the image width and height
		self.imageSize = imageSize
	def decode_and_resize(self, imagePath):
		# read and decode the image path
		image = tf.io.read_file(imagePath)
		image = tf.image.decode_jpeg(image, channels=3)
		# convert the image data type from uint8 to float32 and then resize
		# the image to the set image size
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		image = tf.image.resize(image, self.imageSize)
		# return the image
		return image
	def __call__(self, pair, label):
		positive, negative=pair
		positive = self.decode_and_resize(positive)
		negative = self.decode_and_resize(negative)
		return ( positive, negative), label


class PairGenerator:
    def __init__(self, datasetPath):
        self.fruitNames = list()  # path to dir with fruits
        for folderName in os.listdir(datasetPath):
            absoluteFolderName = os.path.join(datasetPath, folderName)
            numImages = len(os.listdir(absoluteFolderName))
            if numImages > 1:
                self.fruitNames.append(absoluteFolderName)
        self.allFruit = self.generate_all_fruit_dict()

    def generate_all_fruit_dict(self):
        allFruit = dict()

        for fruitName in self.fruitNames:
            imageNames = os.listdir(fruitName)  # all names of photo one fruit
            fruitPhotos = [
                os.path.join(fruitName, imageName) for imageName in imageNames
            ]
            allFruit[fruitName] = fruitPhotos
        return allFruit  # all path photo in dict

    def get_next_element(self):
        i = 0
        while True:
            i = i + 1

            imageNames = random.choice(self.fruitNames)
            temporaryNames = self.fruitNames.copy()
            temporaryNames.remove(imageNames)
            negativeNames = random.choice(temporaryNames)

            imagePhoto = random.choice(self.allFruit[imageNames])
            positivePhoto = random.choice(self.allFruit[imageNames])
            negativePhoto = random.choice(self.allFruit[negativeNames])

            yield ((imagePhoto, positivePhoto), 1)
            yield ((imagePhoto, negativePhoto), 0)

def create_dataset(path, img_size,batch_size):

    AUTO = tf.data.AUTOTUNE
    pair_generator = PairGenerator(path)
    image_processor = MapFunction(img_size)

    dataset = tf.data.Dataset.from_generator(pair_generator.get_next_element,
                                        output_signature=((tf.TensorSpec(shape=(), dtype=tf.string),
                                                            tf.TensorSpec(shape=(), dtype=tf.string)),
                                                            tf.TensorSpec(shape=(), dtype=tf.float32)))
    dataset = dataset.map(image_processor)
    dataset = dataset.batch(batch_size).prefetch(AUTO)
    return dataset
