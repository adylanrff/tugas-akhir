import tensorflow as tf
import numpy as np

class Bilinear(tf.keras.layers.Layer):

    def __init__(self, left_features, right_features, out_features):
        super(Bilinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = self.add_weight(
            name='U',
            shape=(self.out_features, self.left_features, self.right_features),
            trainable=True
        )
        self.W_l = self.add_weight(
            name='W_l',
            shape=(self.left_features, self.out_features),
            trainable=True
        )
        self.W_r = self.add_weight(
            name='W_r',
            shape=(self.right_features, self.out_features),
            trainable=True
        )

        self.bias = self.add_weight(
            name='bias',
            shape=(self.out_features, ),
            trainable=True
        )

    def bilinear(self,x1, x2, A, b):
        x = tf.matmul(x1, A)
        x = tf.transpose(tf.matmul(x, tf.transpose(x2))) + b
        return tf.reduce_sum(x, 0)

    def linear(self,x1, A):
        return tf.matmul(x1, A)

    def call(self, input_left, input_right):
        print("INPUT_LEFT: ", input_left.shape)
        print("INPUT_RIGHT: ", input_right.shape)
        left_size = input_left.shape
        right_size = input_right.shape
        batch = int(np.prod(left_size[:-1]))
        print("BATCH", batch)
        # convert left and right input to matrices [batch, left_features], [batch, right_features]
        input_left = tf.reshape(tf.identity(input_left), shape=(batch, self.left_features))
        input_right = tf.reshape(tf.identity(input_right), shape=(batch, self.right_features))

        # output [batch, out_features]
        output = self.bilinear(input_left, input_right, self.U, self.bias)
        print(output.shape)
        output = output + self.linear(input_left, self.W_l) + self.linear(input_right, self.W_r)
        # convert back to [batch1, batch2, ..., out_features]
        return tf.reshape(output, shape=(left_size[:-1]+[self.out_features]) )
