import keras
from keras import backend as K


class WordCharEmbd(keras.engine.Layer):

    def __init__(self, **kwargs):
        super(WordCharEmbd, self).__init__(**kwargs)

    def build(self, input_shape):
        super(WordCharEmbd, self).build(input_shape)

    def call(self, x, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape
