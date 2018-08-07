
Word/Character Embeddings in Keras
==================================


.. image:: https://travis-ci.org/PoWWoP/keras-word-char-embd.svg
   :target: https://travis-ci.org/PoWWoP/keras-word-char-embd
   :alt: Travis


.. image:: https://coveralls.io/repos/github/PoWWoP/keras-word-char-embd/badge.svg?branch=master
   :target: https://coveralls.io/github/PoWWoP/keras-word-char-embd
   :alt: Coverage


Introduction
------------


.. image:: https://user-images.githubusercontent.com/853842/43352939-c84b9724-925e-11e8-9488-29ef159a69ed.png
   :target: https://user-images.githubusercontent.com/853842/43352939-c84b9724-925e-11e8-9488-29ef159a69ed.png
   :alt: image


Out-of-vocabulary words have bad effects on word embeddings. Sometimes both word and character features are used. The characters in a word are first mapped to character embeddings, then a bidirectional recurrent neural layer is used to encode the character embeddings to a single vector. The final feature of a word is the concatenation of word embedding and encoded character feature.

The repository contains some functions which could be used to generate the first few layers that encodes the features of words.

Install
-------

.. code-block:: bash

   pip install keras-word-char-embd

Demo
^^^^

There is a sentiment analysis demo in the ``demo`` directory. Run the following commands, then your model should have about 70% accuracy:

.. code-block:: bash

   cd demo
   ./get_data.sh
   python sentiment_analysis.py

Functions
^^^^^^^^^

This section only introduces the basic usages of the functions. For more detailed information please refer to the demo and the doc comments describing the functions in the source code.

``get_dicts_generator``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function returns a closure used to generate word and character dictionaries. The closure should be invoked for all the training sentences in order to record the frequencies of each word or character. After that, setting the parameter ``return_dict=True`` the dictionaries would be returned.

.. code-block:: python

   from keras_wc_embd import get_dicts_generator

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

You can generate dictionaries on your own, but make sure index ``0`` and index for ``<UNK>`` are preserved.

``get_embedding_layer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate the first few layers that encodes words in a sentence:

.. code-block:: python

   import keras
   from keras_wc_embd import get_embedding_layer

   inputs, embd_layer = get_embedding_layer(
       word_dict_len=len(word_dict),
       char_dict_len=len(char_dict),
       max_word_len=max_word_len,
       word_embd_dim=300,
       char_embd_dim=50,
       char_hidden_dim=150,
       rnn='lstm',
   )
   model = keras.models.Model(inputs=inputs, outputs=embd_layer)
   model.summary()

The output shape of ``embd_layer`` should be ``(None, None, 600)``\ , which represents the batch size, the length of sentence and the length of encoded word feature.

``get_batch_input``
~~~~~~~~~~~~~~~~~~~~~~~

The function is used to generate the batch inputs for the model.

.. code-block:: python

   from keras_wc_embd import get_batch_input

   word_embd_input, char_embd_input = get_batch_input(
       sentences,
       max_word_len=max_word_len,
       word_dict=word_dict,
       char_dict=char_dict,
   )

``get_embedding_weights_from_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A helper function that loads pre-trained embeddings for initializing the weights of the embedding layer. The format of the file should be similar to GloVe.

.. code-block:: python

   from keras_wc_embd import get_embedding_layer, get_embedding_weights_from_file

   word_embd_weights = get_embedding_weights_from_file(word_dict, 'glove.6B.100d.txt', ignore_case=True)
   inputs, embd_layer = get_embedding_layer(
       word_dict_len=len(word_dict),
       char_dict_len=len(char_dict),
       max_word_len=max_word_len,
       word_embd_dim=300,
       char_embd_dim=50,
       char_hidden_dim=150,
       word_embd_weights=word_embd_weights,
       rnn='lstm',
   )

Citation
^^^^^^^^

Several papers have done the same thing. Just choose the one you have seen.
