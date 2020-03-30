import tensorflow as tf
from tensorflow.keras.layers import concatenate, Embedding, Bidirectional, LSTM
from .glove_embedding import GloveEmbedding

class Encoder(tf.keras.Model):

    NUM_ENCODER_TOKENS = 25
    ENCODER_LATENT_DIM = 200
    OUTPUT_DIM = 100

    def __init__(self, token_vocab_size, pos_vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.token_encoder_embedding = GloveEmbedding(
            token_vocab_size, Encoder.NUM_ENCODER_TOKENS)
        self.pos_encoder_embedding = Embedding(
            input_dim=pos_vocab_size, output_dim=Encoder.OUTPUT_DIM, input_length=Encoder.NUM_ENCODER_TOKENS, mask_zero=True)
        self.bilstm = Bidirectional(LSTM(
            Encoder.ENCODER_LATENT_DIM,
            return_sequences=True,
            return_state=True))

    def call(self, token_input, pos_input, hidden):
        token_encoder_embedding = self.token_encoder_embedding(token_input)
        pos_encoder_embedding = self.pos_encoder_embedding(pos_input)
        x = concatenate(
            [token_encoder_embedding, pos_encoder_embedding])
        output, forward_h, forward_c, backward_h, backward_c = self.bilstm(
            x)
        state_h = concatenate([forward_h, backward_h])
        state_c = concatenate([forward_c, backward_c])
        states = (state_h, state_c)
        return output, states

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))
