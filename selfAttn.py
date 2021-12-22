
import tensorflow as tf
from tensorflow.keras.layers import Layer, Flatten, Activation, Permute
from tensorflow.keras.layers import Permute

class SelfAttn(Layer):
    """
    @param (int) sequence: attention length
    """
    def __init__(self, sequence, return_sequences = True):
        self.sequence = sequence
        self.return_sequences = return_sequences
        super(SelfAttn, self).__init__()

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',
                                  shape=(self.sequence, input_shape[2]),
                                  initializer='random_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(input_shape[1], self.sequence),
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
