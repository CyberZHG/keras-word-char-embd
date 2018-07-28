import unittest
from keras_wc_embd import get_dicts_generator


class TestGetDictsGenerator(unittest.TestCase):

    def test_no_word(self):
        sentences = [
            ['All', 'work', 'and', 'no', 'play'],
            ['makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ]
        dict_generator = get_dicts_generator(
            word_min_freq=2,
            char_min_freq=2,
            word_ignore_case=False,
            char_ignore_case=False,
        )
        for sentence in sentences:
            dict_generator(sentence)
        word_dict, char_dict, max_word_len = dict_generator(return_dict=True)
        self.assertEqual(0, max_word_len)
        self.assertEqual(2, len(word_dict))
        self.assertTrue('u' not in char_dict)
        self.assertTrue('A' not in char_dict)
        self.assertTrue('n' in char_dict)
        self.assertTrue('a' in char_dict)

    def test_ignore_case(self):
        sentences = [
            ['All', 'work', 'and', 'no', 'play', ''],
            ['all', 'worK', 'and', 'no', 'play', '.'],
        ]
        dict_generator = get_dicts_generator(
            word_min_freq=2,
            char_min_freq=2,
            word_ignore_case=True,
            char_ignore_case=True,
        )
        for sentence in sentences:
            dict_generator(sentence)
        word_dict, char_dict, max_word_len = dict_generator(return_dict=True)
        self.assertEqual(4, max_word_len)
        self.assertEqual(7, len(word_dict))
        self.assertTrue('all' in word_dict)
        self.assertTrue('work' in word_dict)
        self.assertTrue('k' in char_dict)
