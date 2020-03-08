import tensorflow as tf
import numpy as np
from .biaffine_attention import BiaffineAttention
from .head_sentinel import HeadSentinel
from tensorflow.keras.layers import Dense, concatenate, Dropout


class DeepBiaffineDecoder(tf.keras.Model):
    def __init__(self):
        super(DeepBiaffineDecoder, self).__init__()
        self.edge_node_h_linear = Dense(256, activation='elu')
        self.edge_node_m_linear = Dense(256, activation='elu')

        self.edge_label_h_linear = Dense(128, activation='elu')
        self.edge_label_m_linear = Dense(128, activation='elu')

        self.head_sentinel = HeadSentinel(400)

        self.encode_dropout = Dropout(0.33)

        self.biaffine_attention = BiaffineAttention(256, 256)

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
            pred_edge_heads, 
            edge_labels, 
            (edge_node_nll + edge_label_nll) / num_nodes, 
            edge_node_nll + edge_label_nll,
            num_nodes)

    def _add_head_sentinel(self, memory_bank, edge_heads, edge_labels, corefs, mask):
        print("MB SHAPE: ", memory_bank.shape)
        batch_size, _, hidden_size = memory_bank.shape
        memory_bank = self.head_sentinel(memory_bank, batch_size, hidden_size)
        print("MEMBANK AFTER SENTINEL: ", memory_bank.shape)
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

        print("EDGE_NODE_H: ", edge_node_h.shape)
        print("EDGE_LABEL_h: ", edge_label_h.shape)
        
        return (edge_node_h, edge_node_m), (edge_label_h, edge_label_m)

    def _get_edge_node_scores(self, edge_node_h, edge_node_m, mask):

        edge_node_scores = tf.squeeze(self.biaffine_attention(
            edge_node_h, edge_node_m),1)
        return edge_node_scores

    def get_loss(self, edge_label_h, edge_label_m, edge_node_scores, edge_heads, edge_labels, mask):
        batch_size, max_len, _ = edge_node_scores.shape

        edge_node_log_likelihood = masked_log_softmax(
            edge_node_scores, mask.unsqueeze(2) + mask.unsqueeze(1), dim=1)

        edge_label_scores = self._get_edge_label_scores(edge_label_h, edge_label_m, edge_heads)
        edge_label_log_likelihood = torch.nn.functional.log_softmax(edge_label_scores, dim=2)

        # Create indexing matrix for batch: [batch, 1]
        batch_index = torch.arange(0, batch_size).view(batch_size, 1).type_as(edge_heads)
        # Create indexing matrix for modifier: [batch, modifier_length]
        modifier_index = torch.arange(0, max_len).view(1, max_len).expand(batch_size, max_len).type_as(edge_heads)
        # Index the log likelihood of gold edges.
        _edge_node_log_likelihood = edge_node_log_likelihood[
            batch_index, edge_heads.data, modifier_index]
        _edge_label_log_likelihood = edge_label_log_likelihood[
            batch_index, modifier_index, edge_labels.data]

        # Exclude the dummy root.
        # Output [batch, length - 1]
        gold_edge_node_nll = - _edge_node_log_likelihood[:, 1:].sum()
        gold_edge_label_nll = - _edge_label_log_likelihood[:, 1:].sum()

        return gold_edge_node_nll, gold_edge_label_nll

