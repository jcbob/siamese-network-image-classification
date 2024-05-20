import tensorflow as tf
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class MapFunction():
    def __init__(self, imageSize):
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
            # Check if the path is a directory and ignore .DS_Store files
            if os.path.isdir(absoluteFolderName) and folderName != '.DS_Store':
                # Filter out .DS_Store files from the image count
                numImages = len([f for f in os.listdir(absoluteFolderName) if f != '.DS_Store' and os.path.isfile(os.path.join(absoluteFolderName, f))])
                if numImages > 1:
                    self.fruitNames.append(absoluteFolderName)
        self.allFruit = self.generate_all_fruit_dict()
        
    def generate_all_fruit_dict(self):
        allFruit = dict()
        for fruitName in self.fruitNames:
            # Ignore .DS_Store files in the directory listing
            imageNames = [f for f in os.listdir(fruitName) if f != '.DS_Store']
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
                a=self.allFruit[anchorName],
                size=2,
                replace=False
            )
            negativePhoto = random.choice(self.allFruit[negativeName])
            yield (anchorPhoto, positivePhoto, negativePhoto)

    def get_next_element_example(self):
        i = 0
        while i < 10:
            i += 1
            anchorName = random.choice(self.fruitNames)
            temporaryNames = self.fruitNames.copy()
            temporaryNames.remove(anchorName)
            negativeName = random.choice(temporaryNames)
            (anchorPhoto, positivePhoto) = np.random.choice(
                a=self.allFruit[anchorName],
                size=2,
                replace=False
            )
            negativePhoto = random.choice(self.allFruit[negativeName])
            yield (anchorPhoto, positivePhoto, negativePhoto)


def load_two_datasets(dataset_path, image_size, batch_size, auto, train_size, sample_size=10000):
    map_fn = MapFunction(image_size)

    print("generating dataset...")
    dataset = tf.data.Dataset.from_generator(
        TripletGenerator(dataset_path).get_next_element,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string)
        )
    )
    print("generated dataset")
    print("type - ", type(dataset))
    print("splitting dataset...")

    # Shuffle and take a sample of the dataset for size estimation
    sample_dataset = dataset.shuffle(sample_size).take(sample_size)
    
    # Estimate total size using the sample
    total_samples = sum(1 for _ in sample_dataset)
    train_samples = int(total_samples * train_size)
    
    # Split dataset
    train_dataset = dataset.take(train_samples)
    val_dataset = dataset.skip(train_samples)
    
    print("split dataset")
    
    train_dataset = train_dataset.map(map_fn)
    train_dataset = train_dataset.batch(batch_size).prefetch(auto)
    
    val_dataset = val_dataset.map(map_fn)
    val_dataset = val_dataset.batch(batch_size).prefetch(auto)

    return train_dataset, val_dataset