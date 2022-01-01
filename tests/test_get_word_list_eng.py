import unittest

from keras_wc_embd import get_word_list_eng


class TestGetWordListEng(unittest.TestCase):

    def test_text(self):
        text = "I'm Jack."
        words = get_word_list_eng(text)
        self.assertEqual(['I', "'", "m", 'Jack', '.'], words)

        text = "  I'm    Jack.  "
        words = get_word_list_eng(text)
        self.assertEqual(['I', "'", "m", 'Jack', '.'], words)

        text = "'m    Jack"
        words = get_word_list_eng(text)
        self.assertEqual(["'", "m", 'Jack'], words)

        text = "    'm    Jack    "
        words = get_word_list_eng(text)
        self.assertEqual(["'", "m", 'Jack'], words)
