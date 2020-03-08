import tensorflow as tf


class BiaffineAttention(tf.keras.layers.Layer):
    def __init__(self, input_size_encoder, input_size_decoder, num_labels=1):
        super(BiaffineAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.W_d = self.add_weight(
            name='W_d',
            shape=(self.num_labels, self.input_size_decoder),
            trainable=True
        )
        self.W_e = self.add_weight(
            name='W_e',
            shape=(self.num_labels, self.input_size_encoder),
            trainable=True
        )
        self.b = self.add_weight(
            name='b',
            shape=(self.num_labels, 1, 1),
            trainable=True
        )
        self.U = self.add_weight(
            name='U',
            shape=(self.num_labels, self.input_size_decoder,
                   self.input_size_encoder),
            trainable=True
        )

    def call(self, input_d, input_e):
        """
        Args:
            input_d: Tensor
                the decoder input tensor with shape = [batch, length_decoder, input_size]
            input_e: Tensor
                the child input tensor with shape = [batch, length_encoder, input_size]
            mask_d: Tensor or None
                the mask tensor for decoder with shape = [batch, length_decoder]
            mask_e: Tensor or None
                the mask tensor for encoder with shape = [batch, length_encoder]
        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]
        """

        batch, length_decoder, _ = input_d.shape
        _, length_encoder, _ = input_e.shape

        # compute decoder part: [num_label, input_size_decoder] * [batch, input_size_decoder, length_decoder]
        # the output shape is [batch, num_label, length_decoder]
        out_d = tf.expand_dims(tf.matmul(self.W_d, tf.transpose(
            input_d, perm=[0, 2, 1])), axis=3)
        print("OUT_D: ", out_d.shape)

        # compute decoder part: [num_label, input_size_encoder] * [batch, input_size_encoder, length_encoder]
        # the output shape is [batch, num_label, length_encoder]
        out_e = tf.expand_dims(tf.matmul(self.W_e, tf.transpose(
            input_d, perm=[0, 2, 1])), axis=3)
        print("OUT_E: ", out_d.shape)

        output = tf.matmul(tf.expand_dims(input_d, axis=1), self.U)
        # [batch, num_label, length_decoder, input_size_encoder] * [batch, 1, input_size_encoder, length_encoder]
        # output shape [batch, num_label, length_decoder, length_encoder]
        output = tf.matmul(output, tf.transpose(tf.expand_dims(input_e, axis=1), [0,1,3,2]))
        
        print("BIAFFINE OUTPUT: ", output.shape)
        return output + out_d + out_e + self.b
        

