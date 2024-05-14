import tensorflow as tf
import numpy as np
import random
import os

class MapFunction():
	def __init__(self, imageSize):
		# define the image width and height
		self.imageSize = imageSize
	def decode_and_resize(self, imagePath):
		image = tf.io.read_file(imagePath)
		image = tf.image.decode_jpeg(image, channels=3)
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		image = tf.image.resize(image, self.imageSize)
		return image
	def __call__(self, anchor, positive, negative):
		anchor = self.decode_and_resize(anchor)
		positive = self.decode_and_resize(positive)
		negative = self.decode_and_resize(negative)
		return (anchor, positive, negative)


class TripletGenerator:
    def __init__(self, datasetPath):
        self.fruitNames = list()
        for folderName in os.listdir(datasetPath):
            absoluteFolderName = os.path.join(datasetPath, folderName)
            numImages = len(os.listdir(absoluteFolderName))
            if numImages > 1:
                self.fruitNames.append(absoluteFolderName)
        self.allFruit = self.generate_all_fruit_dict()
    def generate_all_fruit_dict(self):
        allFruit = dict()
        for fruitName in self.fruitNames:
            imageNames = os.listdir(fruitName)
            fruitPhotos = [
                os.path.join(fruitName, imageName) for imageName in imageNames
                if os.path.isfile(os.path.join(fruitName, imageName))
            ]
            allFruit[fruitName] = fruitPhotos
        return allFruit
    def get_next_element(self):
        while True:
            anchorName = random.choice(self.fruitNames)
            temporaryNames = self.fruitNames.copy()
            temporaryNames.remove(anchorName)
            negativeName = random.choice(temporaryNames)
            (anchorPhoto, positivePhoto) = np.random.choice(
                a = self.allFruit[anchorName],
                size=2,
                replace=False
            )
            negativePhoto = random.choice(self.allFruit[negativeName])
            yield (anchorPhoto, positivePhoto, negativePhoto)
    def get_next_element_example(self):
        i=0
        while i<10:
            i+=1
            anchorName = random.choice(self.fruitNames)
            temporaryNames = self.fruitNames.copy()
            temporaryNames.remove(anchorName)
            negativeName = random.choice(temporaryNames)
            (anchorPhoto, positivePhoto) = np.random.choice(
                a = self.allFruit[anchorName],
                size=2,
                replace=False
            )
            negativePhoto = random.choice(self.allFruit[negativeName])
            yield (anchorPhoto, positivePhoto, negativePhoto)



def load_train_dataset(data_train_path):
    import tensorflow as tf
    train_dataset = tf.data.Dataset.from_generator(
        TripletGenerator(data_train_path).get_next_element,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )
    )
    return train_dataset


def load_test_dataset(data_test_path):
    import tensorflow as tf
    test_dataset = tf.data.Dataset.from_generator(
        TripletGenerator(data_test_path).get_next_element,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )
    )
    return test_dataset

def load_dataset(data_train_path, data_test_path, image_size, batch_size, auto):
    map_fn = MapFunction(image_size)
    
    train_dataset = load_train_dataset(data_train_path)
    train_dataset = train_dataset.map(map_fn)
    train_dataset = train_dataset.batch(batch_size).prefetch(auto)
    
    test_dataset = load_test_dataset(data_test_path)
    test_dataset = test_dataset.map(map_fn)
    test_dataset = test_dataset.batch(batch_size).prefetch(auto)

    return train_dataset, test_dataset