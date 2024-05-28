import tensorflow as tf

def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.keras.backend.sum(tf.keras.backend.square(x - y), axis=1, keepdims=True)
    
    return tf.keras.backend.sqrt(tf.keras.backend.maximum(sum_square, tf.keras.backend.epsilon()))


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

def create_model(embedding_network, input_shape):
    input_1= tf.keras.layers.Input(input_shape)
    input_2= tf.keras.layers.Input(input_shape)
    
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = tf.keras.layers.Lambda(euclidean_distance, output_shape=(1,))(
        [tower_1, tower_2]
    )
    normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(normal_layer)
    
    siamese = tf.keras.Model(inputs=[input_1, input_2], outputs=output_layer)
    return siamese

def loss(margin=1):
    """Provides 'contrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'contrastive_loss' function with data ('margin') attached.
    """

    def contrastive_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)


        square_pred = tf.keras.backend.square(y_pred)
        margin_square = tf.keras.backend.square(tf.keras.backend.maximum(margin - y_pred, 0))
        return tf.keras.backend.mean((1 - y_true) * square_pred + (y_true) * margin_square)


    return contrastive_loss