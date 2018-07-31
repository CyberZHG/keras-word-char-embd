import unittest
import os
from keras_wc_embd import get_dicts_generator, get_embedding_weights_from_file


class TestGetEmbeddingWeightsFromFile(unittest.TestCase):

    def test_load_embd(self):
        sentences = [
            ['All', 'work', 'and', 'no', 'play'],
            ['makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ]
        dict_generator = get_dicts_generator(
            word_min_freq=1,
            char_min_freq=1,
            word_ignore_case=False,
            char_ignore_case=False,
        )
        for sentence in sentences:
            dict_generator(sentence)
        word_dict, char_dict, max_word_len = dict_generator(return_dict=True)

        current = os.path.dirname(os.path.abspath(__file__))
        word_embd_file_path = os.path.join(current, 'demo_word_embd.txt')
        weights = get_embedding_weights_from_file(word_dict, word_embd_file_path, ignore_case=True)
        self.assertEqual((len(word_dict), 3), weights.shape)
        self.assertEqual([0.1, 0.2, 0.3], weights[word_dict['All']].tolist())
        self.assertEqual([0.4, 0.5, 0.6], weights[word_dict['work']].tolist())
        self.assertEqual([0.7, 0.8, 0.9], weights[word_dict['and']].tolist())

        char_embd_file_path = os.path.join(current, 'demo_char_embd.txt')
        weights = get_embedding_weights_from_file(char_dict, char_embd_file_path, ignore_case=True)
        self.assertEqual((len(char_dict), 3), weights.shape)
        self.assertEqual([0.1, 0.2, 0.3], weights[char_dict['A']].tolist())
        self.assertEqual([0.4, 0.5, 0.6], weights[char_dict['l']].tolist())
        self.assertEqual([0.7, 0.8, 0.9], weights[char_dict['w']].tolist())
