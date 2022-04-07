import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import resnet
from itertools import combinations as comb
import tensorflow.keras.backend as K
from config import CV_Config


class FaceResnet:
    def __init__(self, weights_name: str):
        self.target_shape = (150, 150)

        self.siamese_model = None
        self.embedding = None
        self.siamese_network = None

        self.cv_config = CV_Config()
        self.weights_name = weights_name

    def build_model(self):
        base_cnn = resnet.ResNet50(
            weights="imagenet", input_shape=self.target_shape + (3,), include_top=False
        )

        flatten = layers.Flatten()(base_cnn.output)
        dense1 = layers.Dense(512, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        output = layers.Dense(256)(dense2)

        self.embedding = Model(base_cnn.input, output, name="Embedding")

        trainable = False
        for layer in base_cnn.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

        anchor_input = layers.Input(name="anchor", shape=self.target_shape + (3,))
        positive_input = layers.Input(name="positive", shape=self.target_shape + (3,))
        negative_input = layers.Input(name="negative", shape=self.target_shape + (3,))

        distances = DistanceLayer()(
            self.embedding(resnet.preprocess_input(anchor_input)),
            self.embedding(resnet.preprocess_input(positive_input)),
            self.embedding(resnet.preprocess_input(negative_input)),
        )

        siamese_network = Model(
            inputs=[anchor_input, positive_input, negative_input], outputs=distances
        )

        self.siamese_model = SiameseModel(siamese_network)
        self.siamese_model.compile(optimizer=optimizers.Adam(0.0001))
        self.siamese_model.built = True
        self.siamese_model.load_weights(self.cv_config.face_recognition_path + "siamese_network_weights.h5")

        return self.embedding, self.siamese_model


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]