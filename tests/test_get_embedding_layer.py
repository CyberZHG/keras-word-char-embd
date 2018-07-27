import unittest
from keras_wc_embd import get_embedding_layer


class TestGetEmbeddingLayer(unittest.TestCase):

    def test_shape(self):
        inputs, embd_layer = get_embedding_layer(
            word_dict_len=3,
            char_dict_len=5,
            max_word_len=5,
            word_embd_dim=150,
            char_embd_dim=25,
            char_hidden_dim=75,
        )
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0]._keras_shape, (None, None))
        self.assertEqual(inputs[1]._keras_shape, (None, None, 5))
        self.assertEqual(embd_layer._keras_shape, (None, None, 300))
        inputs, embd_layer = get_embedding_layer(
            word_dict_len=3,
            char_dict_len=5,
            max_word_len=7,
            word_embd_dim=300,
            char_embd_dim=50,
            char_hidden_dim=150,
            rnn='gru',
        )
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0]._keras_shape, (None, None))
        self.assertEqual(inputs[1]._keras_shape, (None, None, 7))
        self.assertEqual(embd_layer._keras_shape, (None, None, 600))
