import keras
from keras.models import load_model
import tensorflow as tf


class ObjectAssociationModel:
    def __init__(self, path):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            with self.session.as_default():
                self.model = load_model(path)
                self.graph = tf.get_default_graph()

    def predict(self, img1, img2):
        with self.graph.as_default():
            with self.session.as_default():
                output = self.model.predict([img1, img2], verbose=0)
                return output
