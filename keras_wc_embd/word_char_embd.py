import codecs
import numpy
from .backend import keras
from .backend import backend as K

__all__ = [
    'MaskedConv1D', 'MaskedFlatten',
    'get_batch_input', 'get_embedding_layer', 'get_dicts_generator',
    'get_word_list_eng', 'get_embedding_weights_from_file',
]


class MaskedConv1D(keras.layers.Conv1D):

    def __init__(self, **kwargs):
        super(MaskedConv1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskedConv1D, self).call(inputs)


class MaskedFlatten(keras.layers.Flatten):

    def __init__(self, **kwargs):
        super(MaskedFlatten, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask


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

    return [numpy.asarray(word_embd_input), numpy.asarray(char_embd_input)]


def get_embedding_layer(word_dict_len,
                        char_dict_len,
                        max_word_len,
                        word_embd_dim=300,
                        char_embd_dim=30,
                        char_hidden_dim=150,
                        char_hidden_layer_type='lstm',
                        word_embd_weights=None,
                        char_embd_weights=None,
                        word_embd_trainable=None,
                        char_embd_trainable=None,
                        word_mask_zero=True,
                        char_mask_zero=True):
    """Get the merged embedding layer.

    :param word_dict_len: The number of words in the dictionary including the ones mapped to 0 or 1.
    :param char_dict_len: The number of characters in the dictionary including the ones mapped to 0 or 1.
    :param max_word_len: The maximum allowed length of word.
    :param word_embd_dim: The dimensions of the word embedding.
    :param char_embd_dim: The dimensions of the character embedding
    :param char_hidden_dim: The dimensions of the hidden states of RNN in one direction.
    :param word_embd_weights: A numpy array representing the pre-trained embeddings for words.
    :param char_embd_weights: A numpy array representing the pre-trained embeddings for characters.
    :param word_embd_trainable: Whether the word embedding layer is trainable.
    :param char_embd_trainable: Whether the character embedding layer is trainable.
    :param char_hidden_layer_type: The type of the recurrent layer, 'lstm' or 'gru'.
    :param word_mask_zero: Whether enable the mask for words.
    :param char_mask_zero: Whether enable the mask for characters.

    :return inputs, embd_layer: The keras layer.
    """
    if word_embd_weights is not None:
        word_embd_weights = [word_embd_weights]
    if word_embd_trainable is None:
        word_embd_trainable = word_embd_weights is None

    if char_embd_weights is not None:
        char_embd_weights = [char_embd_weights]
    if char_embd_trainable is None:
        char_embd_trainable = char_embd_weights is None

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
        mask_zero=word_mask_zero,
        weights=word_embd_weights,
        trainable=word_embd_trainable,
        name='Embedding_Word',
    )(word_input_layer)
    char_embd_layer = keras.layers.Embedding(
        input_dim=char_dict_len,
        output_dim=char_embd_dim,
        mask_zero=char_mask_zero,
        weights=char_embd_weights,
        trainable=char_embd_trainable,
        name='Embedding_Char_Pre',
    )(char_input_layer)
    if char_hidden_layer_type == 'lstm':
        char_hidden_layer = keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=char_hidden_dim,
                input_shape=(max_word_len, char_dict_len),
                return_sequences=False,
                return_state=False,
            ),
            name='Bi-LSTM_Char',
        )
    elif char_hidden_layer_type == 'gru':
        char_hidden_layer = keras.layers.Bidirectional(
            keras.layers.GRU(
                units=char_hidden_dim,
                input_shape=(max_word_len, char_dict_len),
                return_sequences=False,
                return_state=False,
            ),
            name='Bi-GRU_Char',
        )
    elif char_hidden_layer_type == 'cnn':
        char_hidden_layer = [
            MaskedConv1D(
                filters=max(1, char_hidden_dim // 5),
                kernel_size=3,
                activation='relu',
            ),
            MaskedFlatten(),
            keras.layers.Dense(
                units=char_hidden_dim,
                name='Dense_Char',
            ),
        ]
    elif isinstance(char_hidden_layer_type, list) or isinstance(char_hidden_layer_type, keras.layers.Layer):
        char_hidden_layer = char_hidden_layer_type
    else:
        raise NotImplementedError('Unknown character hidden layer type: %s' % char_hidden_layer_type)
    if not isinstance(char_hidden_layer, list):
        char_hidden_layer = [char_hidden_layer]
    for i, layer in enumerate(char_hidden_layer):
        if i == len(char_hidden_layer) - 1:
            name = 'Embedding_Char'
        else:
            name = 'Embedding_Char_Pre_%d' % (i + 1)
        char_embd_layer = keras.layers.TimeDistributed(layer=layer, name=name)(char_embd_layer)
    embd_layer = keras.layers.Concatenate(
        name='Embedding',
    )([word_embd_layer, char_embd_layer])
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
    """A naive function that extracts English words from raw texts.

    :param text: The raw text.
    :return words: A list of strings.
    """
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


def get_embedding_weights_from_file(word_dict, file_path, ignore_case=False):
    """Load pre-trained embeddings from a text file.

    Each line in the file should look like this:
        word feature_dim_1 feature_dim_2 ... feature_dim_n

    The `feature_dim_i` should be a floating point number.

    :param word_dict: A dict that maps words to indice.
    :param file_path: The location of the text file containing the pre-trained embeddings.
    :param ignore_case: Whether ignoring the case of the words.

    :return weights: A numpy array.
    """
    pre_trained = {}
    with codecs.open(file_path, 'r', 'utf8') as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if ignore_case:
                parts[0] = parts[0].lower()
            pre_trained[parts[0]] = list(map(float, parts[1:]))
    embd_dim = len(next(iter(pre_trained.values())))
    weights = [[0.0] * embd_dim for _ in range(max(word_dict.values()) + 1)]
    for word, index in word_dict.items():
        if not word:
            continue
        if ignore_case:
            word = word.lower()
        if word in pre_trained:
            weights[index] = pre_trained[word]
        else:
            weights[index] = numpy.random.random((embd_dim,)).tolist()
    return numpy.asarray(weights)
