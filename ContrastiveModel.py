import tensorflow as tf

def create_embedding(img_shape):
    input = tf.keras.layers.Input(shape=img_shape)
    x = tf.keras.layers.BatchNormalization()(input)
    x = tf.keras.layers.Conv2D(4, (5, 5), activation="tanh")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (5, 5), activation="tanh")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(100, activation="tanh")(x)
    embedding_network = tf.keras.Model(input, x)

    return embedding_network