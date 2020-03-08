import tensorflow as tf
from tensorflow.keras.layers import concatenate

class HeadSentinel(tf.keras.layers.Layer):
    def __init__(self, input_size):
        super(HeadSentinel, self).__init__()
        self.sentinel = self.add_weight(
            name='sentinel',
            shape=(1,1,input_size)
        )

    def call(self, memory_bank, batch_size, hidden_size):
        head_sentinel = tf.broadcast_to(self.sentinel, [batch_size, 1, hidden_size])
        memory_bank = concatenate([head_sentinel, memory_bank], axis=1)
        return memory_bank