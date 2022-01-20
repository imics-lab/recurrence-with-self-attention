"""
Self-Attendion Layer

Authors: Alexander Katrompas, Theodoros Ntakouris, Vangelis Metsis
Organization: Texas State University

Stand alone self-attendion layer class for use with LSTM layer, conforming to the
transformer concept from Attention Is All You Need (Vaswani 2017)
https://arxiv.org/abs/1706.03762

"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Flatten, Activation, Permute
from tensorflow.keras.layers import Permute

class SelfAttn(Layer):
    """
    Stand alone self-attendion layer class for use with LSTM layer, conforming
    to the transformer concept from Attention Is All You Need (Vaswani 2017)
    https://arxiv.org/abs/1706.03762

    @param (int) alength: attention length
    @param (bool) return_sequences: return sequences true (default) or false
    """
    def __init__(self, alength, return_sequences = True):
        self.alength = alength
        self.return_sequences = return_sequences
        super(SelfAttn, self).__init__()

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',
                                  shape=(self.alength, input_shape[2]),
                                  initializer='random_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(input_shape[1], self.alength),
                                  initializer='random_uniform',
                                  trainable=True)
        super(SelfAttn, self).build(input_shape)

    def call(self, inputs):
        W1, W2 = self.W1[None, :, :], self.W2[None, :, :]
        hidden_states_transposed = Permute(dims=(2, 1))(inputs)
        attention_score = tf.matmul(W1, hidden_states_transposed)
        attention_score = Activation('tanh')(attention_score)
        attention_weights = tf.matmul(W2, attention_score)
        attention_weights = Activation('softmax')(attention_weights)
        embedding_matrix = tf.matmul(attention_weights, inputs)
        if not self.return_sequences:
            embedding_matrix = Flatten()(embedding_matrix)
        return embedding_matrix
