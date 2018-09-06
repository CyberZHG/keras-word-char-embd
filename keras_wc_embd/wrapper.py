from .word_char_embd import (get_batch_input,
                             get_embedding_layer,
                             get_dicts_generator,
                             get_embedding_weights_from_file)


class WordCharEmbd(object):

    def __init__(self,
                 word_min_freq=4,
                 char_min_freq=2,
                 word_ignore_case=False,
                 char_ignore_case=False):
        """Wrapper initialization

        :param word_min_freq: The minimum frequency of a word.
        :param char_min_freq: The minimum frequency of a character.
        :param word_ignore_case: Word will be transformed to lower case before saving to dictionary.
        :param char_ignore_case: Character will be transformed to lower case before saving to dictionary.
        """
        self.word_ignore_case = word_ignore_case
        self.char_ignore_case = char_ignore_case
        self.max_word_len = 5
        self.dict_generator = get_dicts_generator(word_min_freq=word_min_freq,
                                                  char_min_freq=char_min_freq,
                                                  word_ignore_case=word_ignore_case,
                                                  char_ignore_case=char_ignore_case)
        self.word_dict, self.char_dict = None, None

    def update_dicts(self, sentence):
        """Add new sentence to generate dictionaries.

        :param sentence: A list of strings representing the sentence.
        """
        self.dict_generator(sentence=sentence)
        self.word_dict, self.char_dict = None, None

    def get_dicts(self):
        """Get word and character dictionaries.

        :return word_dict, char_dict:
        """
        if self.word_dict is None:
            self.word_dict, self.char_dict, self.max_word_len = self.dict_generator(return_dict=True)
        return self.word_dict, self.char_dict

    def get_word_dicts(self):
        """Get the word dictionary.

        :return word_dict:
        """
        word_dict, _ = self.get_dicts()
        return word_dict

    def get_char_dicts(self):
        """Get the character dictionary.

        :return char_dict:
        """
        _, char_dict = self.get_dicts()
        return char_dict

    def get_embedding_layer(self,
                            word_embd_dim=300,
                            char_embd_dim=30,
                            char_hidden_dim=150,
                            rnn='lstm',
                            word_embd_weights=None,
                            word_embd_file_path=None,
                            char_embd_weights=None,
                            word_embd_trainable=None,
                            char_embd_trainable=None,
                            mask_zeros=True):
        """Get the merged embedding layer.

        :param word_embd_dim: The dimensions of the word embedding.
        :param char_embd_dim: The dimensions of the character embedding
        :param char_hidden_dim: The dimensions of the hidden states of RNN in one direction.
        :param word_embd_weights: A numpy array representing the pre-trained embeddings for words.
        :param word_embd_file_path: The file that contains the word embeddings.
        :param char_embd_weights: A numpy array representing the pre-trained embeddings for characters.
        :param word_embd_trainable: Whether the word embedding layer is trainable.
        :param char_embd_trainable: Whether the character embedding layer is trainable.
        :param rnn: The type of the recurrent layer, 'lstm' or 'gru'.
        :param mask_zeros: Whether enable the mask.

        :return inputs, embd_layer: The keras layer.
        """
        if word_embd_file_path is not None:
            word_embd_weights = get_embedding_weights_from_file(word_dict=self.word_dict,
                                                                file_path=word_embd_file_path,
                                                                ignore_case=self.word_ignore_case)
        return get_embedding_layer(word_dict_len=len(self.word_dict),
                                   char_dict_len=len(self.char_dict),
                                   max_word_len=self.max_word_len,
                                   word_embd_dim=word_embd_dim,
                                   char_embd_dim=char_embd_dim,
                                   char_hidden_dim=char_hidden_dim,
                                   rnn=rnn,
                                   word_embd_weights=word_embd_weights,
                                   char_embd_weights=char_embd_weights,
                                   word_embd_trainable=word_embd_trainable,
                                   char_embd_trainable=char_embd_trainable,
                                   mask_zeros=mask_zeros)

    def get_batch_input(self, sentences):
        """Convert sentences to desired input tensors.

        :param sentences: A list of lists representing the input sentences.

        :return word_embd_input, char_embd_input: The desired inputs.
        """
        return get_batch_input(sentences,
                               max_word_len=self.max_word_len,
                               word_dict=self.word_dict,
                               char_dict=self.char_dict,
                               word_ignore_case=self.word_ignore_case,
                               char_ignore_case=self.char_ignore_case)
