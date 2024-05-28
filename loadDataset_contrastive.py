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

    def __call__(self, pair, label):
        positive, negative = pair
        positive = self.decode_and_resize(positive)
        negative = self.decode_and_resize(negative)
        return (positive, negative), label


class PairGenerator:
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

    def _generate_label_images_dict(self):
        label_images = {}
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
            val_images.extend(images[num_train:num_train + num_val])
            test_images.extend(images[num_train + num_val:])

        return train_images, val_images, test_images

    def _get_pair(self, image_set):
        while True:
            positive_image = random.choice(image_set)
            negative_image = random.choice(image_set)

            label_positive = os.path.dirname(positive_image)
            label_negative = os.path.dirname(negative_image)

            if label_positive == label_negative:
                yield (positive_image, negative_image), 1
            else:
                yield (positive_image, negative_image), 0

    def get_train_element(self):
        return self._get_pair(self.train_images)

    def get_val_element(self):
        return self._get_pair(self.val_images)

    def get_test_element(self):
        return self._get_pair(self.test_images)





def create_dataset(path, img_size, batch_size):
    pair_generator = PairGenerator(path)
    image_processor = MapFunction(img_size)

    train_dataset = tf.data.Dataset.from_generator(pair_generator.get_train_element,
                                                   output_signature=((tf.TensorSpec(shape=(), dtype=tf.string),
                                                                      tf.TensorSpec(shape=(), dtype=tf.string)),
                                                                     tf.TensorSpec(shape=(), dtype=tf.int32)))

    val_dataset = tf.data.Dataset.from_generator(pair_generator.get_val_element,
                                                 output_signature=((tf.TensorSpec(shape=(), dtype=tf.string),
                                                                    tf.TensorSpec(shape=(), dtype=tf.string)),
                                                                   tf.TensorSpec(shape=(), dtype=tf.int32)))

    test_dataset = tf.data.Dataset.from_generator(pair_generator.get_test_element,
                                                  output_signature=((tf.TensorSpec(shape=(), dtype=tf.string),
                                                                     tf.TensorSpec(shape=(), dtype=tf.string)),
                                                                    tf.TensorSpec(shape=(), dtype=tf.int32)))

    train_dataset = train_dataset.map(image_processor).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(image_processor).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(image_processor).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset



# import tensorflow as tf
# import numpy as np
# import random
# import os

# class MapFunction():
# 	def __init__(self, imageSize):
# 		# define the image width and height
# 		self.imageSize = imageSize
# 	def decode_and_resize(self, imagePath):
# 		# read and decode the image path
# 		image = tf.io.read_file(imagePath)
# 		image = tf.image.decode_jpeg(image, channels=3)
# 		# convert the image data type from uint8 to float32 and then resize
# 		# the image to the set image size
# 		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
# 		image = tf.image.resize(image, self.imageSize)
# 		# return the image
# 		return image
# 	def __call__(self, pair, label):
# 		positive, negative=pair
# 		positive = self.decode_and_resize(positive)
# 		negative = self.decode_and_resize(negative)
# 		return ( positive, negative), label


# class PairGenerator:
#     def __init__(self, datasetPath):
#         self.fruitNames = list()  # path to dir with fruits
#         for folderName in os.listdir(datasetPath):
#             absoluteFolderName = os.path.join(datasetPath, folderName)
#             numImages = len(os.listdir(absoluteFolderName))
#             if numImages > 1:
#                 self.fruitNames.append(absoluteFolderName)
#         self.allFruit = self.generate_all_fruit_dict()

#     def generate_all_fruit_dict(self):
#         allFruit = dict()

#         for fruitName in self.fruitNames:
#             imageNames = os.listdir(fruitName)  # all names of photo one fruit
#             fruitPhotos = [
#                 os.path.join(fruitName, imageName) for imageName in imageNames
#             ]
#             allFruit[fruitName] = fruitPhotos
#         return allFruit  # all path photo in dict

#     def get_next_element(self):
#         i = 0
#         while True:
#             i = i + 1

#             imageNames = random.choice(self.fruitNames)
#             temporaryNames = self.fruitNames.copy()
#             temporaryNames.remove(imageNames)
#             negativeNames = random.choice(temporaryNames)

#             imagePhoto = random.choice(self.allFruit[imageNames])
#             positivePhoto = random.choice(self.allFruit[imageNames])
#             negativePhoto = random.choice(self.allFruit[negativeNames])

#             yield ((imagePhoto, positivePhoto), 1)
#             yield ((imagePhoto, negativePhoto), 0)

# def create_dataset(path, img_size,batch_size):

#     AUTO = tf.data.AUTOTUNE
#     pair_generator = PairGenerator(path)
#     image_processor = MapFunction(img_size)

#     dataset = tf.data.Dataset.from_generator(pair_generator.get_next_element,
#                                         output_signature=((tf.TensorSpec(shape=(), dtype=tf.string),
#                                                             tf.TensorSpec(shape=(), dtype=tf.string)),
#                                                             tf.TensorSpec(shape=(), dtype=tf.float32)))
#     dataset = dataset.map(image_processor)
#     dataset = dataset.batch(batch_size).prefetch(AUTO)
#     return dataset
