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
            max_word_len=5,
            word_dict={},
            char_dict={},
        )
        self.assertEqual(word_embd_input.shape, (2, 6))
        self.assertEqual(char_embd_input.shape, (2, 6, 5))
        for sentence_index in range(2):
            for word_index in range(6):
                if word_index < len(sentences[sentence_index]):
                    self.assertEqual(
                        1,
                        word_embd_input[sentence_index, word_index],
                        (sentence_index, word_index),
                    )
                    for char_index in range(5):
                        if char_index < len(sentences[sentence_index][word_index]):
                            self.assertEqual(
                                1,
                                char_embd_input[sentence_index, word_index, char_index].tolist(),
                                (sentence_index, word_index),
                            )
                        else:
                            self.assertEqual(
                                0,
                                char_embd_input[sentence_index, word_index, char_index].tolist(),
                                (sentence_index, word_index),
                            )
                else:
                    self.assertEqual(
                        0,
                        word_embd_input[sentence_index, word_index],
                        (sentence_index, word_index),
                    )

    def test_mapping(self):
        sentences = [
            ['All', 'work', 'and', 'no', 'play'],
            ['makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ]
        word_embd_input, char_embd_input = get_batch_input(
            sentences,
            max_word_len=5,
            word_dict={'All': 2, 'Work': 0},
            char_dict={'a': 3},
            word_ignore_case=False,
            char_ignore_case=False,
        )
        self.assertEqual(word_embd_input.shape, (2, 6))
        self.assertEqual(char_embd_input.shape, (2, 6, 5))
        for sentence_index in range(2):
            for word_index in range(6):
                if word_index < len(sentences[sentence_index]):
                    if sentences[sentence_index][word_index] == 'All':
                        self.assertEqual(
                            2,
                            word_embd_input[sentence_index, word_index],
                            (sentence_index, word_index),
                        )
                    else:
                        self.assertEqual(
                            1,
                            word_embd_input[sentence_index, word_index],
                            (sentence_index, word_index),
                        )
                    for char_index in range(5):
                        if char_index < len(sentences[sentence_index][word_index]):
                            if sentences[sentence_index][word_index][char_index] == 'a':
                                self.assertEqual(
                                    3,
                                    char_embd_input[sentence_index, word_index, char_index].tolist(),
                                    (sentence_index, word_index),
                                )
                            else:
                                self.assertEqual(
                                    1,
                                    char_embd_input[sentence_index, word_index, char_index].tolist(),
                                    (sentence_index, word_index),
                                )
                        else:
                            self.assertEqual(
                                0,
                                char_embd_input[sentence_index, word_index, char_index].tolist(),
                                (sentence_index, word_index),
                            )
                else:
                    self.assertEqual(
                        0,
                        word_embd_input[sentence_index, word_index],
                        (sentence_index, word_index),
                    )

    def test_ignore_case(self):
        sentences = [
            ['All', 'work', 'and', 'no', 'play'],
            ['makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ]
        word_embd_input, char_embd_input = get_batch_input(
            sentences,
            max_word_len=5,
            word_dict={'all': 2, 'Work': 0},
            char_dict={'a': 3},
            word_ignore_case=True,
            char_ignore_case=True,
        )
        self.assertEqual(word_embd_input.shape, (2, 6))
        self.assertEqual(char_embd_input.shape, (2, 6, 5))
        for sentence_index in range(2):
            for word_index in range(6):
                if word_index < len(sentences[sentence_index]):
                    if sentences[sentence_index][word_index] == 'All':
                        self.assertEqual(
                            2,
                            word_embd_input[sentence_index, word_index],
                            (sentence_index, word_index),
                        )
                    else:
                        self.assertEqual(
                            1,
                            word_embd_input[sentence_index, word_index],
                            (sentence_index, word_index),
                        )
                    for char_index in range(5):
                        if char_index < len(sentences[sentence_index][word_index]):
                            if sentences[sentence_index][word_index][char_index].lower() == 'a':
                                self.assertEqual(
                                    3,
                                    char_embd_input[sentence_index, word_index, char_index].tolist(),
                                    (sentence_index, word_index),
                                )
                            else:
                                self.assertEqual(
                                    1,
                                    char_embd_input[sentence_index, word_index, char_index].tolist(),
                                    (sentence_index, word_index),
                                )
                        else:
                            self.assertEqual(
                                0,
                                char_embd_input[sentence_index, word_index, char_index].tolist(),
                                (sentence_index, word_index),
                            )
                else:
                    self.assertEqual(
                        0,
                        word_embd_input[sentence_index, word_index],
                        (sentence_index, word_index),
                    )

    def test_exceed_len(self):
        sentences = [
            ['All', 'work', 'and', 'no', 'play'],
            ['makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ]
        word_embd_input, char_embd_input = get_batch_input(
            sentences,
            max_word_len=2,
            word_dict={},
            char_dict={},
        )
        self.assertEqual(word_embd_input.shape, (2, 6))
        self.assertEqual(char_embd_input.shape, (2, 6, 2))
