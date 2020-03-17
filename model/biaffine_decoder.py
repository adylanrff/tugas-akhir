import tensorflow as tf
import torch
import numpy as np
from .biaffine_attention import BiaffineAttention
from .head_sentinel import HeadSentinel
from .bilinear import Bilinear
from tensorflow.keras.layers import Dense, concatenate, Dropout, Lambda
from .util import create_indices, create_edge_label_index, create_edge_node_index

class DeepBiaffineDecoder(tf.keras.Model):
    def __init__(self, vocab):
        super(DeepBiaffineDecoder, self).__init__()
        self.edge_node_h_linear = Dense(256, activation='elu')
        self.edge_node_m_linear = Dense(256, activation='elu')

        edge_label_hidden_size = 128
        self.edge_label_h_linear = Dense(edge_label_hidden_size, activation='elu')
        self.edge_label_m_linear = Dense(edge_label_hidden_size, activation='elu')

        self.head_sentinel = HeadSentinel(400)

        self.encode_dropout = Dropout(0.33)

        self.biaffine_attention = BiaffineAttention(256, 256)

        num_labels = vocab.get_vocab_size("head_tags")
        self.edge_label_bilinear = Bilinear(edge_label_hidden_size, edge_label_hidden_size, num_labels)
        
        self.minus_inf = -1e8
        
    def call(self, memory_bank, edge_heads, edge_labels, corefs, mask):
        num_nodes = tf.keras.backend.sum(mask)
        memory_bank, edge_heads, edge_labels, corefs, mask = self._add_head_sentinel(
            memory_bank, edge_heads, edge_labels, corefs, mask)

        (edge_node_h, edge_node_m), (edge_label_h,
                                     edge_label_m) = self.encode(memory_bank)

        edge_node_scores = self._get_edge_node_scores(
            edge_node_h, edge_node_m, mask)

        edge_node_nll, edge_label_nll = self.get_loss(
            edge_label_h, edge_label_m, edge_node_scores, edge_heads, edge_labels, mask)

        pred_edge_heads, pred_edge_labels = self.decode(
            edge_label_h, edge_label_m, edge_node_scores, corefs, mask)

        return ( 
            tf.cast(pred_edge_heads, dtype='float32'), 
            tf.cast(edge_labels, dtype='float32'),
        ) 
            # (edge_node_nll + edge_label_nll) / num_nodes, 
            # edge_node_nll + edge_label_nll,
            # num_nodes)

    def _add_head_sentinel(self, memory_bank, edge_heads, edge_labels, corefs, mask):
        batch_size, _, hidden_size = memory_bank.shape
        memory_bank = self.head_sentinel(memory_bank, batch_size, hidden_size)
        if edge_heads is not None:
            edge_heads = concatenate([tf.zeros((batch_size, 1), dtype='int32'), edge_heads], 1)
        if edge_labels is not None:
            edge_labels = concatenate([tf.zeros((batch_size, 1), dtype='int32'), edge_labels], 1)
        if corefs is not None:
            corefs = concatenate([tf.zeros((batch_size, 1), dtype='int32'), corefs], 1)
        mask = concatenate([tf.zeros((batch_size, 1), dtype='int32'), mask], 1)
        return memory_bank, edge_heads, edge_labels, corefs, mask


    def encode(self, memory_bank):
        # Output: [batch, length, edge_hidden_size]
        edge_node_h = self.edge_node_h_linear(memory_bank)
        edge_node_m = self.edge_node_m_linear(memory_bank)

        # Output: [batch, length, label_hidden_size]
        edge_label_h = self.edge_label_h_linear(memory_bank)
        edge_label_m = self.edge_label_m_linear(memory_bank)

        edge_node = concatenate([edge_node_h, edge_node_m], axis=1)
        edge_label = concatenate([edge_label_h, edge_label_m], axis=1)

        edge_node = self.encode_dropout(edge_node)
        edge_label = self.encode_dropout(edge_label)
        
        edge_node_h, edge_node_m = tf.split(edge_node, 2, 1)
        edge_label_h, edge_label_m = tf.split(edge_label, 2, 1)

        return (edge_node_h, edge_node_m), (edge_label_h, edge_label_m)

    def _get_edge_node_scores(self, edge_node_h, edge_node_m, mask):

        edge_node_scores = tf.squeeze(self.biaffine_attention(
            edge_node_h, edge_node_m),1)
        return edge_node_scores

    def _get_edge_label_scores(self, edge_label_h, edge_label_m, edge_heads):
        """
        Compute the edge label scores.
        :param edge_label_h: [batch, length, edge_label_hidden_size]
        :param edge_label_m: [batch, length, edge_label_hidden_size]
        :param heads: [batch, length] -- element at [i, j] means the head index of node_j at batch_i.
        :return: [batch, length, num_labels]
        """

        indices = create_indices(edge_heads)
        edge_label_h = tf.gather_nd(edge_label_h, indices)
        # [batch, length, num_labels]
        edge_label_scores = self.edge_label_bilinear(edge_label_h, edge_label_m)

        return edge_label_scores

    def masked_log_softmax(self, vector, mask, dim=-1):
        if mask is not None:
            mask = tf.cast(mask, dtype='float32')
            while len(mask.shape) < len(vector.shape):
                mask = tf.expand_dims(mask, 1)
            # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
            # results in nans when the whole vector is masked.  We need a very small value instead of a
            # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
            # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
            # becomes 0 - this is just the smallest value we can actually use.
            vector = vector + tf.keras.backend.log((mask + 1e-45))
        return tf.nn.log_softmax(vector, axis=dim)

    def get_loss(self, edge_label_h, edge_label_m, edge_node_scores, edge_heads, edge_labels, mask):
        batch_size, max_len, _ = edge_node_scores.shape

        edge_node_log_likelihood = self.masked_log_softmax(
            edge_node_scores, tf.expand_dims(mask, 2) + tf.expand_dims(mask, 1), dim=1)
        edge_label_scores = self._get_edge_label_scores(edge_label_h, edge_label_m, edge_heads)
        edge_label_log_likelihood = tf.nn.log_softmax(edge_label_scores, axis=2)

        # Create indexing matrix for batch: [batch, 1]
        batch_index = tf.range(batch_size)
        modifier_index = tf.range(max_len)

        edge_node_indices = create_edge_node_index(batch_index, edge_heads, modifier_index)
        edge_label_indices = create_edge_label_index(batch_index, modifier_index, edge_labels)
        # Create indexing matrix for modifier: [batch, modifier_length]
        # Index the log likelihood of gold edges.

        _edge_node_log_likelihood = tf.gather_nd(edge_node_log_likelihood, edge_node_indices)
        _edge_label_log_likelihood = tf.gather_nd(edge_label_log_likelihood, edge_label_indices)

        # Exclude the dummy root.
        # Output [batch, length - 1]
        gold_edge_node_nll = - tf.math.reduce_sum(_edge_node_log_likelihood[:, 1:])
        gold_edge_label_nll = - tf.math.reduce_sum(_edge_label_log_likelihood[:, 1:])
        return gold_edge_node_nll, gold_edge_label_nll

    def decode(self,edge_label_h, edge_label_m, edge_node_scores, corefs, mask):

        max_len = edge_node_scores.shape[1]

        # Set diagonal elements to -inf
        edge_node_scores = edge_node_scores + tf.linalg.diag(tf.fill([max_len], -np.inf))

        # Set invalid positions to -inf
        minus_mask = (1 - tf.cast(mask, dtype='float32')) * self.minus_inf
        edge_node_scores = edge_node_scores + tf.expand_dims(minus_mask, 2) + tf.expand_dims(minus_mask, 1)

        # Compute naive predictions.
        # prediction shape = [batch, length]
        edge_heads = tf.keras.backend.argmax(edge_node_scores, axis=1)
        # Based on predicted heads, compute the edge label scores.
        # [batch, length, num_labels]
        edge_label_scores = self._get_edge_label_scores(edge_label_h, edge_label_m, edge_heads)
        edge_labels = tf.keras.backend.max(edge_label_scores, axis=2)

        return edge_heads[:, 1:], edge_labels[:, 1:]