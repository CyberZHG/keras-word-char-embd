import keras
import numpy


def get_batch_input(sentences,
                    max_word_len,
                    word_dict,
                    char_dict,
                    word_unknown=1,
                    char_unknown=1,
                    word_ignore_case=False,
                    char_ignore_case=False):
    """Convert sentences to desired input tensors.

    :param sentences: A list of lists representing the input sentences.
    :param max_word_len: The maximum allowed length of word.
    :param word_dict: Map a word to an integer. (0 and 1 should be preserved)
    :param char_dict: Map a character to an integer. (0 and 1 should be preserved)
    :param word_unknown: An integer representing the unknown word.
    :param char_unknown: An integer representing the unknown character.
    :param word_ignore_case: Word will be transformed to lower case before mapping.
    :param char_ignore_case: Character will be transformed to lower case before mapping.

    :return word_embd_input, char_embd_input: The desired inputs.
    """
    sentence_num = len(sentences)

    max_sentence_len = max(map(len, sentences))

    word_embd_input = [[0] * max_sentence_len for _ in range(sentence_num)]
    char_embd_input = [[[0] * max_word_len for _ in range(max_sentence_len)] for _ in range(sentence_num)]

    for sentence_index, sentence in enumerate(sentences):
        for word_index, word in enumerate(sentence):
            if word_ignore_case:
                word_key = word.lower()
            else:
                word_key = word
            word_embd_input[sentence_index][word_index] = word_dict.get(word_key, word_unknown)
            for char_index, char in enumerate(word):
                if char_index >= max_word_len:
                    break
                if char_ignore_case:
                    char = char.lower()
                char_embd_input[sentence_index][word_index][char_index] = char_dict.get(char, char_unknown)

    return numpy.asarray(word_embd_input), numpy.asarray(char_embd_input)


def get_embedding_layer(word_dict_len,
                        char_dict_len,
                        max_word_len,
                        word_embd_dim=300,
                        char_embd_dim=30,
                        char_hidden_dim=150,
                        rnn='lstm',
                        mask_zeros=True):
    """Get the merged embedding layer.

    :param word_dict_len: The number of words in the dictionary including the ones mapped to 0 or 1.
    :param char_dict_len: The number of characters in the dictionary including the ones mapped to 0 or 1.
    :param max_word_len: The maximum allowed length of word.
    :param word_embd_dim: The dimensions of the word embedding.
    :param char_embd_dim: The dimensions of the character embedding
    :param char_hidden_dim: The dimensions of the hidden states of RNN in one direction.
    :param rnn: The type of the recurrent layer, 'lstm' or 'gru'.
    :param mask_zeros: Whether enable the mask.

    :return inputs, embd_layer: The keras layer.
    """
    word_input_layer = keras.layers.Input(
        shape=(None,),
        name='Input_Word',
    )
    char_input_layer = keras.layers.Input(
        shape=(None, max_word_len),
        name='Input_Char',
    )

    word_embd_layer = keras.layers.Embedding(
        input_dim=word_dict_len,
        output_dim=word_embd_dim,
        mask_zero=mask_zeros,
        name='Embedding_Word',
    )(word_input_layer)
    char_embd_layer = keras.layers.Embedding(
        input_dim=char_dict_len,
        output_dim=char_embd_dim,
        mask_zero=mask_zeros,
        name='Embedding_Char_Pre',
    )(char_input_layer)
    if rnn == 'lstm':
        char_rnn_layer = keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=char_hidden_dim,
                input_shape=(max_word_len, char_dict_len),
                return_sequences=False,
                return_state=False,
            ),
            name='Bi-LSTM_Char',
        )
    else:
        char_rnn_layer = keras.layers.Bidirectional(
            keras.layers.GRU(
                units=char_hidden_dim,
                input_shape=(max_word_len, char_dict_len),
                return_sequences=False,
                return_state=False,
            ),
            name='Bi-GRU_Char',
        )
    char_embd_layer = keras.layers.TimeDistributed(
        layer=char_rnn_layer,
        name='Embedding_Char'
    )(char_embd_layer)
    embd_layer = keras.layers.concatenate(
        inputs=[word_embd_layer, char_embd_layer],
        name='Embedding',
    )
    return [word_input_layer, char_input_layer], embd_layer


def get_dicts_generator(word_min_freq=4,
                        char_min_freq=2,
                        word_ignore_case=False,
                        char_ignore_case=False):
    """Get word and character dictionaries from sentences.

    :param word_min_freq: The minimum frequency of a word.
    :param char_min_freq: The minimum frequency of a character.
    :param word_ignore_case: Word will be transformed to lower case before saving to dictionary.
    :param char_ignore_case: Character will be transformed to lower case before saving to dictionary.
    :return gen: A closure that accepts sentences and returns the dictionaries.
    """
    word_count, char_count = {}, {}

    def get_dicts(sentence=None,
                  return_dict=False):
        """Update and return dictionaries for each sentence.

        :param sentence: A list of strings representing the sentence.
        :param return_dict: Returns the dictionaries if it is True.
        :return word_dict, char_dict, max_word_len:
        """
        if sentence is not None:
            for word in sentence:
                if not word:
                    continue
                if word_ignore_case:
                    word_key = word.lower()
                else:
                    word_key = word
                word_count[word_key] = word_count.get(word_key, 0) + 1
                for char in word:
                    if char_ignore_case:
                        char_key = char.lower()
                    else:
                        char_key = char
                    char_count[char_key] = char_count.get(char_key, 0) + 1
        if not return_dict:
            return None
        word_dict, char_dict = {'': 0, '<UNK>': 1}, {'': 0, '<UNK>': 1}
        max_word_len = 0
        for word, count in word_count.items():
            if count >= word_min_freq:
                word_dict[word] = len(word_dict)
                max_word_len = max(max_word_len, len(word))
        for char, count in char_count.items():
            if count >= char_min_freq:
                char_dict[char] = len(char_dict)
        return word_dict, char_dict, max_word_len

    return get_dicts


def get_word_list_eng(text):
    words, index = [''], 0
    while index < len(text):
        while index < len(text) and ('a' <= text[index] <= 'z' or 'A' <= text[index] <= 'Z'):
            words[-1] += text[index]
            index += 1
        if words[-1]:
            words.append('')
        while index < len(text) and not ('a' <= text[index] <= 'z' or 'A' <= text[index] <= 'Z'):
            if text[index] != ' ':
                words[-1] += text[index]
            index += 1
        if words[-1]:
            words.append('')
    if not words[-1]:
        words.pop()
    return words
