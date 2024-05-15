import tensorflow as tf
from keras.applications import resnet
from keras import layers
from keras import Model


class DistanceLayer(layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = self.square_distance(anchor, positive)
        an_distance = self.square_distance(anchor, negative)
        return (ap_distance, an_distance)

    def square_distance(self, x, y):
        return tf.reduce_sum(tf.square(x - y), axis=-1)


def generate_siamese_triplet_network(target_shape):
    base_cnn = resnet.ResNet50(
        weights="imagenet", input_shape=target_shape + (3,), include_top=False
    )
    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(dense2)

    embedding = Model(base_cnn.input, output, name="Embedding")

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    anchor_input = layers.Input(name = "anchor", shape = target_shape + (3,))
    positive_input = layers.Input(name = "positive", shape = target_shape + (3,))
    negative_input = layers.Input(name = "negative", shape = target_shape + (3,))

    distances = DistanceLayer()(
        embedding(resnet.preprocess_input(anchor_input)),
        embedding(resnet.preprocess_input(positive_input)),
        embedding(resnet.preprocess_input(negative_input)),
    )
    
    siamese_network = Model(
        inputs = [anchor_input, positive_input, negative_input],
        outputs = distances
    )
    
    return siamese_network