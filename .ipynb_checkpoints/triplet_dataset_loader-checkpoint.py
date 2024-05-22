import tensorflow as tf
import numpy as np
import random
import os

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
    def __init__(self, datasetPath, split_ratio=(0.7, 0.2, 0.1)):
        self.datasetPath = datasetPath
        self.split_ratio = split_ratio
        self.label_names = self._get_label_names()
        self.label_images = self._generate_label_images_dict()
        self.train_images, self.val_images, self.test_images = self._split_label_images()

    def _get_label_names(self):
        label_names = []
        for folder_name in os.listdir(self.datasetPath):
            folder_path = os.path.join(self.datasetPath, folder_name)
            if os.path.isdir(folder_path):
                subfolders = [os.path.join(folder_name, subfolder) for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]
                if subfolders:
                    label_names.extend(subfolders)
                else:
                    label_names.append(folder_name)
        return label_names

    def show_label_names(self):
        for label in self.label_names:
            print(label)

    def show_label_images_dict(self):
        for label, images in self.label_images.items():
            print(f"{label} - {images}")

        
    def _generate_label_images_dict(self):
        label_images = dict()
        for label_name in self.label_names:
            label_path = os.path.join(self.datasetPath, label_name)
            if os.path.isdir(label_path):
                image_files = [os.path.join(label_path, imageName) for imageName in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, imageName))]
            else:
                image_files = [os.path.join(self.datasetPath, label_name, imageName) for imageName in os.listdir(os.path.join(self.datasetPath, label_name)) if os.path.isfile(os.path.join(self.datasetPath, label_name, imageName))]
            label_images[label_name] = image_files
        return label_images

    def _split_label_images(self):
        train_images = []
        val_images = []
        test_images = []

        for label, images in self.label_images.items():
            random.shuffle(images)
            num_train = int(len(images) * self.split_ratio[0])
            num_val = int(len(images) * self.split_ratio[1])
            
            train_images.extend(images[:num_train])
            val_images.extend(images[num_train:num_train+num_val])
            test_images.extend(images[num_train+num_val:])

        return train_images, val_images, test_images
    
    def _get_triplet(self, image_list):
        while True:
            anchor_image = random.choice(image_list)
            label_anchor = os.path.dirname(anchor_image)

            positive_image = random.choice([
                img for img in image_list
                if os.path.dirname(img) == label_anchor and img != anchor_image
            ])
            
            negative_image = random.choice([
                img for img in image_list
                if os.path.dirname(img) != label_anchor
            ])

            yield(anchor_image, positive_image, negative_image)

    def get_train_element(self):
        return self._get_triplet(self.train_images)

    def get_val_element(self):
        return self._get_triplet(self.val_images)

    def get_test_element(self):
        return self._get_triplet(self.test_images)


def create_dataset(path, img_size, batch_size):
    triplet_generator = TripletGeneratorNew(path)
    image_processor = MapFunction(img_size)

    train_dataset = tf.data.Dataset.from_generator(triplet_generator.get_train_element,
                                                   output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                                                                     tf.TensorSpec(shape=(), dtype=tf.string),
                                                                     tf.TensorSpec(shape=(), dtype=tf.string)))

    val_dataset = tf.data.Dataset.from_generator(triplet_generator.get_val_element,
                                                 output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                                                                   tf.TensorSpec(shape=(), dtype=tf.string),
                                                                   tf.TensorSpec(shape=(), dtype=tf.string)))

    test_dataset = tf.data.Dataset.from_generator(triplet_generator.get_test_element,
                                                  output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                                                                    tf.TensorSpec(shape=(), dtype=tf.string),
                                                                    tf.TensorSpec(shape=(), dtype=tf.string)))

    train_dataset = train_dataset.map(image_processor).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(image_processor).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(image_processor).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset