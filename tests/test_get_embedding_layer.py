import unittest
import numpy
import keras
from keras_wc_embd import get_embedding_layer


class TestGetEmbeddingLayer(unittest.TestCase):

    def test_shape_rnn(self):
        inputs, embd_layer = get_embedding_layer(
            word_dict_len=3,
            char_dict_len=5,
            max_word_len=5,
            word_embd_dim=150,
            char_embd_dim=25,
            char_hidden_dim=75,
        )
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0].shape.as_list(), [None, None])
        self.assertEqual(inputs[1].shape.as_list(), [None, None, 5])
        self.assertEqual(embd_layer.shape.as_list(), [None, None, 300])
        inputs, embd_layer = get_embedding_layer(
            word_dict_len=3,
            char_dict_len=5,
            max_word_len=7,
            word_embd_dim=300,
            char_embd_dim=50,
            char_hidden_dim=150,
            char_hidden_layer_type='gru',
        )
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0].shape.as_list(), [None, None])
        self.assertEqual(inputs[1].shape.as_list(), [None, None, 7])
        self.assertEqual(embd_layer.shape.as_list(), [None, None, 600])

    def test_pre_trained_shape(self):
        get_embedding_layer(
            word_dict_len=3,
            char_dict_len=5,
            max_word_len=5,
            word_embd_dim=150,
            char_embd_dim=25,
            char_hidden_dim=75,
            word_embd_weights=numpy.random.random((3, 150)),
            char_embd_weights=numpy.random.random((5, 25)),
        )

        def word_embd_wrong_shape():
            get_embedding_layer(
                word_dict_len=3,
                char_dict_len=5,
                max_word_len=5,
                word_embd_dim=150,
                char_embd_dim=25,
                char_hidden_dim=75,
                word_embd_weights=numpy.random.random((3, 100)),
                char_embd_weights=numpy.random.random((5, 25)),
            )

        self.assertRaises(ValueError, word_embd_wrong_shape)

        def char_embd_wrong_shape():
            get_embedding_layer(
                word_dict_len=3,
                char_dict_len=5,
                max_word_len=5,
                word_embd_dim=150,
                char_embd_dim=25,
                char_hidden_dim=75,
                word_embd_weights=numpy.random.random((3, 150)),
                char_embd_weights=numpy.random.random((7, 25)),
            )

        self.assertRaises(ValueError, char_embd_wrong_shape)

    def test_shape_cnn(self):
        inputs, embd_layer = get_embedding_layer(
            word_dict_len=3,
            char_dict_len=5,
            max_word_len=7,
            word_embd_dim=300,
            char_embd_dim=50,
            char_hidden_dim=150,
            char_hidden_layer_type='cnn',
            char_mask_zero=True,
        )
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0].shape.as_list(), [None, None])
        self.assertEqual(inputs[1].shape.as_list(), [None, None, 7])
        self.assertEqual(embd_layer.shape.as_list(), [None, None, 450])

    def test_custom(self):
        layers = [
            keras.layers.Conv1D(
                filters=16,
                kernel_size=3,
                activation='relu',
            ),
            keras.layers.Conv1D(
                filters=16,
                kernel_size=3,
                activation='relu',
            ),
            keras.layers.Flatten(),
            keras.layers.Dense(
                units=50,
                name='Dense_Char',
            ),
        ]
        inputs, embd_layer = get_embedding_layer(
            word_dict_len=3,
            char_dict_len=5,
            max_word_len=7,
            word_embd_dim=300,
            char_embd_dim=50,
            char_hidden_layer_type=layers,
            char_mask_zero=False,
        )
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0].shape.as_list(), [None, None])
        self.assertEqual(inputs[1].shape.as_list(), [None, None, 7])
        self.assertEqual(embd_layer.shape.as_list(), [None, None, 350])
        inputs, embd_layer = get_embedding_layer(
            word_dict_len=3,
            char_dict_len=5,
            max_word_len=7,
            word_embd_dim=300,
            char_embd_dim=50,
            char_hidden_layer_type=keras.layers.GRU(units=30),
            char_mask_zero=False,
        )
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0].shape.as_list(), [None, None])
        self.assertEqual(inputs[1].shape.as_list(), [None, None, 7])
        self.assertEqual(embd_layer.shape.as_list(), [None, None, 330])

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            get_embedding_layer(
                word_dict_len=3,
                char_dict_len=5,
                max_word_len=7,
                char_hidden_layer_type='Jack',
            )
