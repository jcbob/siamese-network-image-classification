import tensorflow as tf
import numpy as np
import random
import os

class MapFunction():
    def __init__(self, imageSize):
        self.imageSize = imageSize

    def decode_and_resize(self, imagePath):
        image = tf.io.read_file(imagePath)
        try:
            image = tf.image.decode_jpeg(image, channels=3)
        except tf.errors.InvalidArgumentError:
            try:
                image = tf.image.decode_image(image, channels=3)
            except tf.errors.InvalidArgumentError:
                raise ValueError(f"Unsupported image format for file: {imagePath}")

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
        self._clean_ds_store_files()
        self.split_ratio = split_ratio
        self.label_names = self._get_label_names()
        self.label_images_dict = self._generate_label_images_dict()
        self.train_images, self.val_images, self.test_images_dict = self._split_label_images()

    def _clean_ds_store_files(self):
        # Remove .DS_Store files from the datasetPath
        for root, _, files in os.walk(self.datasetPath):
            for file in files:
                if file == '.DS_Store':
                    os.remove(os.path.join(root, file))
                    # print(f"Removed .DS_Store file: {os.path.join(root, file)}")

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

    def return_label_names_list(self):
        return self.label_names

    def return_label_images_dict(self):
        return self.label_images_dict

    def _generate_label_images_dict(self):
        label_images_dict = dict()
        for label_name in self.label_names:
            label_path = os.path.join(self.datasetPath, label_name)
            if os.path.isdir(label_path):
                image_files = [os.path.join(label_path, imageName) for imageName in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, imageName))]
            else:
                image_files = [os.path.join(self.datasetPath, label_name, imageName) for imageName in os.listdir(os.path.join(self.datasetPath, label_name)) if os.path.isfile(os.path.join(self.datasetPath, label_name, imageName))]
            label_images_dict[label_name] = image_files
        return label_images_dict

    def _split_label_images(self):
        train_images = []
        val_images = []
        test_images_dict = {}
    
        for label, images in self.label_images_dict.items():
            # Shuffle images and labels together
            combined = list(zip(images, [label] * len(images)))
            random.shuffle(combined)
            shuffled_images, shuffled_labels = zip(*combined)
    
            # Calculate split indices based on split_ratio
            num_train = int(len(shuffled_images) * self.split_ratio[0])
            num_val = int(len(shuffled_images) * self.split_ratio[1])
            
            # Extend train_images and val_images
            train_images.extend(shuffled_images[:num_train])
            val_images.extend(shuffled_images[num_train:num_train + num_val])
            
            # Populate test_images_dict
            test_images_dict.update({img_path: label for img_path, label in zip(shuffled_images[num_train + num_val:], shuffled_labels[num_train + num_val:])})
    
        return train_images, val_images, test_images_dict
    
    def _get_triplet(self, image_list):
        while True:
            anchor_image = random.choice(image_list)
            label_anchor = os.path.dirname(anchor_image)
    
            positive_candidates = [
                img for img in image_list
                if os.path.dirname(img) == label_anchor and img != anchor_image
            ]
    
            if not positive_candidates:
                print(f"No positive candidates for anchor image: {anchor_image} with label: {label_anchor}")
                continue
    
            positive_image = random.choice(positive_candidates)
            
            negative_candidates = [
                img for img in image_list
                if os.path.dirname(img) != label_anchor
            ]
    
            if not negative_candidates:
                print(f"No negative candidates for anchor image: {anchor_image} with label: {label_anchor}")
                continue
    
            negative_image = random.choice(negative_candidates)
    
            yield (anchor_image, positive_image, negative_image)

    def get_train_element(self):
        return self._get_triplet(self.train_images)

    def get_val_element(self):
        return self._get_triplet(self.val_images)

    # def get_test_element(self):
    #     return self._get_triplet(list(self.test_images_dict.keys()))
    
    #     def get_test_element(self):
    #         return self._get_triplet(self.test_images)

    def get_test_images_dict(self):
        return self.test_images_dict

    # def test_triplet_generation(self, num_triplets=5):
    #     print("Testing triplet generation:")
    #     triplet_generator = self.get_test_element()
    #     for _ in range(num_triplets):
    #         try:
    #             anchor, positive, negative = next(triplet_generator)
    #             print(f"Anchor: {anchor}, Positive: {positive}, Negative: {negative}")
    #         except StopIteration:
    #             print("No more triplets available.")
    #             break











def create_dataset(path, split_ratio, img_size, batch_size):
    triplet_generator = TripletGenerator(path, split_ratio)
    image_processor = MapFunction(img_size)

    train_dataset = tf.data.Dataset.from_generator(triplet_generator.get_train_element,
                                                   output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                                                                     tf.TensorSpec(shape=(), dtype=tf.string),
                                                                     tf.TensorSpec(shape=(), dtype=tf.string)))

    val_dataset = tf.data.Dataset.from_generator(triplet_generator.get_val_element,
                                                 output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                                                                   tf.TensorSpec(shape=(), dtype=tf.string),
                                                                   tf.TensorSpec(shape=(), dtype=tf.string)))

    # test_dataset = tf.data.Dataset.from_generator(triplet_generator.get_test_element,
    #                                               output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
    #                                                                 tf.TensorSpec(shape=(), dtype=tf.string),
    #                                                                 tf.TensorSpec(shape=(), dtype=tf.string)))
    test_images_dict = triplet_generator.get_test_images_dict()
    test_dataset = test_images_dict
    # test_dataset = {img_path: triplet_generator.label_images_dict[label] for img_path, label in test_images_dict.items()}

    train_dataset = train_dataset.map(image_processor).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(image_processor).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # test_dataset = test_dataset.map(image_processor).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset













    



# class TripletGenerator:
#     def __init__(self, datasetPath, split_ratio=(0.7, 0.2, 0.1)):
#         self.datasetPath = datasetPath
#         self.split_ratio = split_ratio
#         self.label_names = self._get_label_names()
#         self.label_images_dict = self._generate_label_images_dict()
#         self.train_images, self.val_images, self.test_images = self._split_label_images()

#     def _get_label_names(self):
#         label_names = []
#         for folder_name in os.listdir(self.datasetPath):
#             folder_path = os.path.join(self.datasetPath, folder_name)
#             if os.path.isdir(folder_path):
#                 subfolders = [os.path.join(folder_name, subfolder) for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]
#                 if subfolders:
#                     label_names.extend(subfolders)
#                 else:
#                     label_names.append(folder_name)
#         return label_names

#     def return_label_names_list(self):
#         # for label in self.label_names:
#         #     print(label)
#         return self.label_names

#     def return_label_images_dict(self):
#         # for label, images in self.label_images_dict.items():
#         #     print(f"{label} - {images}")
#         return self.label_images_dict

#     def _generate_label_images_dict(self):
#         label_images_dict = dict()
#         for label_name in self.label_names:
#             label_path = os.path.join(self.datasetPath, label_name)
#             if os.path.isdir(label_path):
#                 image_files = [os.path.join(label_path, imageName) for imageName in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, imageName))]
#             else:
#                 image_files = [os.path.join(self.datasetPath, label_name, imageName) for imageName in os.listdir(os.path.join(self.datasetPath, label_name)) if os.path.isfile(os.path.join(self.datasetPath, label_name, imageName))]
#             label_images_dict[label_name] = image_files
#         return label_images_dict

#     def _split_label_images(self):
#         train_images = []
#         val_images = []
#         test_images = []

#         for label, images in self.label_images_dict.items():
#             random.shuffle(images)
#             num_train = int(len(images) * self.split_ratio[0])
#             num_val = int(len(images) * self.split_ratio[1])
            
#             train_images.extend(images[:num_train])
#             val_images.extend(images[num_train:num_train+num_val])
#             test_images.extend(images[num_train+num_val:])

#         return train_images, val_images, test_images
    
#     # def _get_triplet(self, image_list):
#     #     while True:
#     #         anchor_image = random.choice(image_list)
#     #         label_anchor = os.path.dirname(anchor_image)

#     #         positive_image = random.choice([
#     #             img for img in image_list
#     #             if os.path.dirname(img) == label_anchor and img != anchor_image
#     #         ])
            
#     #         negative_image = random.choice([
#     #             img for img in image_list
#     #             if os.path.dirname(img) != label_anchor
#     #         ])

#     #         yield(anchor_image, positive_image, negative_image)

#     def _get_triplet(self, image_list):
#         while True:
#             anchor_image = random.choice(image_list)
#             label_anchor = os.path.dirname(anchor_image)
    
#             positive_candidates = [
#                 img for img in image_list
#                 if os.path.dirname(img) == label_anchor and img != anchor_image
#             ]
    
#             if not positive_candidates:
#                 print(f"No positive candidates for anchor image: {anchor_image} with label: {label_anchor}")
#                 continue
    
#             positive_image = random.choice(positive_candidates)
            
#             negative_candidates = [
#                 img for img in image_list
#                 if os.path.dirname(img) != label_anchor
#             ]
    
#             if not negative_candidates:
#                 print(f"No negative candidates for anchor image: {anchor_image} with label: {label_anchor}")
#                 continue
    
#             negative_image = random.choice(negative_candidates)
    
#             # Debug prints
#             # print(f"Anchor image: {anchor_image}")
#             # print(f"Positive image: {positive_image}")
#             # print(f"Negative image: {negative_image}")
    
#             yield(anchor_image, positive_image, negative_image)

    
#     def get_train_element(self):
#         return self._get_triplet(self.train_images)

#     def get_val_element(self):
#         return self._get_triplet(self.val_images)



#     def get_test_images_dict(self):
#         return self.test_images

#     def test_triplet_generation(self, num_triplets=5):
#             print("Testing triplet generation:")
#             triplet_generator = self.get_test_element()
#             for _ in range(num_triplets):
#                 try:
#                     anchor, positive, negative = next(triplet_generator)
#                     print(f"Anchor: {anchor}, Positive: {positive}, Negative: {negative}")
#                 except StopIteration:
#                     print("No more triplets available.")
#                     break


