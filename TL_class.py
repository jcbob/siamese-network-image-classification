import tensorflow as tf
from keras import Model
from keras import metrics

class SiameseTripletModel(Model):
    """The Siamese Network model with custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        self.accuracy_tracker = metrics.Mean(name="accuracy")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss, accuracy = self._compute_loss_and_accuracy(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_tracker.result()}

    def test_step(self, data):
        loss, accuracy = self._compute_loss_and_accuracy(data)

        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_tracker.result()}

    def _compute_loss_and_accuracy(self, data):
        ap_distance, an_distance = self.siamese_network(data)

        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)

        accuracy = tf.cast(ap_distance + self.margin < an_distance, tf.float32)
        accuracy = tf.reduce_mean(accuracy)

        return loss, accuracy

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]

        
# import tensorflow as tf
# from keras import Model
# from keras import metrics

# class SiameseTripletModel(Model):
#     """The Siamese Network model with custom training and testing loops.

#     Computes the triplet loss using the three embeddings produced by the
#     Siamese Network.

#     The triplet loss is defined as:
#        L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
#     """

#     def __init__(self, siamese_network, margin=0.5):
#         super().__init__()
#         self.siamese_network = siamese_network
#         self.margin = margin
#         self.loss_tracker = metrics.Mean(name="loss")

#     def call(self, inputs):
#         return self.siamese_network(inputs)

#     def train_step(self, data):
#         # GradientTape is a context manager that records every operation that
#         # you do inside. We are using it here to compute the loss so we can get
#         # the gradients and apply them using the optimizer specified in
#         # `compile()`.
#         with tf.GradientTape() as tape:
#             loss = self._compute_loss(data)

#         # Storing the gradients of the loss function with respect to the
#         # weights/parameters.
#         gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

#         # Applying the gradients on the model using the specified optimizer
#         self.optimizer.apply_gradients(
#             zip(gradients, self.siamese_network.trainable_weights)
#         )

#         # Let's update and return the training loss metric.
#         self.loss_tracker.update_state(loss)
#         return {"loss": self.loss_tracker.result()}

#     def test_step(self, data):
#         loss = self._compute_loss(data)

#         # Let's update and return the loss metric.
#         self.loss_tracker.update_state(loss)
#         return {"loss": self.loss_tracker.result()}

#     def _compute_loss(self, data):
#         # The output of the network is a tuple containing the distances
#         # between the anchor and the positive example, and the anchor and
#         # the negative example.
#         ap_distance, an_distance = self.siamese_network(data)

#         # Computing the Triplet Loss by subtracting both distances and
#         # making sure we don't get a negative value.
#         loss = ap_distance - an_distance
#         loss = tf.maximum(loss + self.margin, 0.0)
#         return loss

#     @property
#     def metrics(self):
#         # We need to list our metrics here so the `reset_states()` can be
#         # called automatically.
#         return [self.loss_tracker]