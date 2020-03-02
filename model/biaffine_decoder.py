import tensorflow as tf
from .biaffine_attention import BiaffineAttention
from tensorflow.keras.layers import Dense

class DeepBiaffineDecoder(tf.keras.Model):
    def __init__(self):
        super(DeepBiaffineDecoder, self).__init__()
        self.edge_node_h_linear = Dense(256)
        self.edge_node_m_linear = Dense(256)
        
        self.edge_label_h_linear = Dense(128)
        self.edge_label_m_linear = Dense(128)

        self.biaffine_attention = BiaffineAttention(256, 256)

    def call(self, memory_bank, edge_heads, edge_labels, corefs):

        memory_bank, edge_heads, edge_labels, corefs, mask = self._add_head_sentinel(
            memory_bank, edge_heads, edge_labels, corefs, mask)

        (edge_node_h, edge_node_m), (edge_label_h, edge_label_m) = self.encode(memory_bank)

        edge_node_scores = self._get_edge_node_scores(edge_node_h, edge_node_m, mask)

        edge_node_nll, edge_label_nll = self.get_loss(
            edge_label_h, edge_label_m, edge_node_scores, edge_heads, edge_labels, mask)

        pred_edge_heads, pred_edge_labels = self.decode(
            edge_label_h, edge_label_m, edge_node_scores, corefs, mask)
        
        
        return True

    def _add_head_sentinel(self):
        pass

    def encode(self, memory_bank):
        pass

    def _get_edge_node_scores(self, edge_node_h, edge_node_m, mask):

        edge_node_scores = self.biaffine_attention(edge_node_h, edge_node_m, mask_d=mask, mask_e=mask).squeeze(1)
        return edge_node_scores
        pass