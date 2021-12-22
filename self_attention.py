import tensorflow.keras as keras

class SelfAttentionBlock(keras.Model):
    def __init__(self, name='SelfAttentionBlock', num_heads=2, head_size=128, ff_dim=None, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size, dropout=dropout)
        self.attention_dropout = keras.layers.Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = keras.layers.Conv1D(
            filters=ff_dim, kernel_size=1, activation='relu')
        # self.ff_conv2 at build()
        self.ff_dropout = keras.layers.Dropout(dropout)
        self.ff_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_conv2 = keras.layers.Conv1D(
            filters=input_shape[-1], kernel_size=1)

    def call(self, inputs, training, **kwargs):
        x = self.attention(inputs, inputs, **kwargs)
        x = self.attention_dropout(x, training=training, **kwargs)
        x = self.attention_norm(inputs + x, **kwargs)

        x = self.ff_conv1(x, **kwargs)
        x = self.ff_conv2(x, **kwargs)
        x = self.ff_dropout(x, training=training, **kwargs)

        x = self.ff_norm(inputs + x, **kwargs)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
