import tensorflow.keras as keras
from self_attention import SelfAttentionBlock


class TransformerEncoder(keras.Model):
    def __init__(self, name='TransformerEncoder', num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0.1, **kwargs):
        super().__init__(name=name, **kwargs)
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.attention_layers = [SelfAttentionBlock(
            num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]

    def call(self, inputs, training, **kwargs):
        x = inputs
        for attention_layer in self.attention_layers:
            x = attention_layer(x, training, **kwargs)

        return x
