import keras
import numpy


def get_batch_input(sentences,
                    word_dict,
                    char_dict,
                    word_dict_len,
                    char_dict_len,
                    word_unknown=1,
                    char_unknown=1,
                    ignore_word_case=False,
                    ignore_char_case=False):
    """Convert sentences to desired input tensors.

    :param sentences: A list of lists representing the input sentences.
    :param word_dict: Map a word to an integer. (0 and 1 should be preserved)
    :param char_dict: Map a character to an integer. (0 and 1 should be preserved)
    :param word_dict_len: The number of words in the dictionary including the ones mapped to 0 or 1.
    :param char_dict_len: The number of characters in the dictionary including the ones mapped to 0 or 1.
    :param word_unknown: An integer representing the unknown word.
    :param char_unknown: An integer representing the unknown character.
    :param ignore_word_case: Word will be transformed to lower case before mapping.
    :param ignore_char_case: Character will be transformed to lower case before mapping.

    :return word_embd_input, char_embd_input: The desired inputs.
    """
    sentence_num = len(sentences)

    max_sentence_len = max(map(len, sentences))
    max_word_len = max(map(lambda sentence: max(map(len, sentence)), sentences))

    word_embd_input = [
        [
            [0.0] * word_dict_len
            for _ in range(max_sentence_len)
        ]
        for _ in range(sentence_num)
    ]
    char_embd_input = [
        [
            [
                [0.0] * char_dict_len
                for _ in range(max_word_len)
            ]
            for _ in range(max_sentence_len)
        ]
        for _ in range(sentence_num)
    ]

    for sentence_index, sentence in enumerate(sentences):
        for word_index, word in enumerate(sentence):
            if ignore_word_case:
                word_key = word.lower()
            else:
                word_key = word
            word_pos = word_dict.get(word_key, word_unknown)
            word_embd_input[sentence_index][word_index][word_pos] = 1.0
            for char_index, char in enumerate(word):
                if ignore_char_case:
                    char = char.lower()
                char_pos = char_dict.get(char, char_unknown)
                char_embd_input[sentence_index][word_index][char_index][char_pos] = 1.0

    return numpy.asarray(word_embd_input), numpy.asarray(char_embd_input)
