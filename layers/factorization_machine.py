import tensorflow as tf
from tensorflow import keras


class FactorizationMachine(keras.layers.Layer):
    def __init__(self, units=32,
                 embed_dim=5,
                 kernel_initializer='glorot_uniform',
                 embedding_initializer = 'uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 embedding_regularizer=None,
                 bias_regularizer=None):
        super(FactorizationMachine, self).__init__()
        self.units = units
        self.embed_dim = embed_dim
        self.kernel_initializer = kernel_initializer
        self.embedding_initializer = embedding_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.embedding_regularizer = embedding_regularizer

    def build(self, input_shape):
        self.v = self.add_weight(
            shape=(input_shape[-1], self.embed_dim, self.units),
            initializer=self.embedding_initializer,
            regularizer=self.embedding_regularizer,
            trainable=True,
        )

    def call(self, inputs):
        einsum_equation = 'bei,ieo->beo'

        broadcast_shape = [self.embed_dim, tf.shape(inputs)[0], tf.shape(inputs)[1]]
        x = tf.broadcast_to(inputs, broadcast_shape)
        # x.shape should be [embed_dim, batch_num, input_dim]

        x = tf.transpose(x, perm=[1, 0, 2])
        # x.shape should be [batch_num, embed_dim, input_dim]

        first_term = tf.math.square(tf.einsum(einsum_equation, x, self.v))
        # the shape of the first term should be [batch_num, embed_dim, output units]

        second_term = tf.einsum(einsum_equation, tf.math.square(x), tf.math.square(self.v))

        output = tf.reduce_sum(tf.math.subtract(first_term, second_term), 1)
        return output
