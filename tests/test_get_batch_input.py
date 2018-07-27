import unittest
from keras_wc_embd import get_batch_input


class TestGetBatchInput(unittest.TestCase):

    def test_shape(self):
        sentences = [
            ['All', 'work', 'and', 'no', 'play'],
            ['makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ]
        word_embd_input, char_embd_input = get_batch_input(
            sentences,
            word_dict={},
            char_dict={},
            word_dict_len=3,
            char_dict_len=7
        )
        self.assertEqual(word_embd_input.shape, (2, 6, 3))
        self.assertEqual(char_embd_input.shape, (2, 6, 5, 7))
