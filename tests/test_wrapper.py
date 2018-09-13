import unittest
import os
import keras
import numpy as np
from keras_wc_embd import WordCharEmbd


class TestWrapper(unittest.TestCase):

    def test_wrapper(self):
        wc_embd = WordCharEmbd(word_min_freq=1,
                               char_min_freq=1,
                               word_ignore_case=True,
                               char_ignore_case=True)
        sentences = [
            ['All', 'work', 'and', 'no', 'play'],
            ['makes', 'Jack', 'a', 'dull', 'boy', '.'],
        ]
        for sentence in sentences:
            wc_embd.update_dicts(sentence)
        word_dict = wc_embd.get_word_dicts()
        char_dict = wc_embd.get_char_dicts()
        wc_embd.set_dicts(word_dict, char_dict)
        current = os.path.dirname(os.path.abspath(__file__))
        word_embd_file_path = os.path.join(current, 'demo_word_embd.txt')
        char_embd_file_path = os.path.join(current, 'demo_char_embd.txt')
        inputs, embd_layer = wc_embd.get_embedding_layer(
            word_embd_dim=3,
            char_embd_dim=3,
            word_embd_file_path=word_embd_file_path,
            char_embd_file_path=char_embd_file_path,
            word_embd_trainable=True,
            char_embd_trainable=True,
        )
        lstm_layer = keras.layers.LSTM(units=5, name='LSTM')(embd_layer)
        softmax_layer = keras.layers.Dense(units=2, activation='softmax', name='Softmax')(lstm_layer)
        model = keras.models.Model(inputs=inputs, outputs=softmax_layer)
        model.compile(
            optimizer='adam',
            loss=keras.losses.sparse_categorical_crossentropy,
            metrics=[keras.metrics.sparse_categorical_accuracy],
        )
        model.summary()

        def batch_generator():
            while True:
                yield wc_embd.get_batch_input(sentences * 50), np.asarray([0, 1] * 50)

        model.fit_generator(
            generator=batch_generator(),
            steps_per_epoch=200,
            epochs=1,
        )
        predicts = model.predict(wc_embd.get_batch_input(sentences))
        self.assertGreater(predicts[0][0], predicts[0][1])
        self.assertLess(predicts[1][0], predicts[1][1])
