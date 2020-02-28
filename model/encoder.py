import tensorflow as tf

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    token_encoder_embedding = GloveEmbedding(self.encoder_token_vocab_size, TextToAMR.NUM_ENCODER_TOKENS)(token_encoder_input)
    pos_encoder_embedding = Embedding(input_dim=self.encoder_pos_vocab_size,output_dim=100, input_length=TextToAMR.NUM_ENCODER_TOKENS)(pos_encoder_input)
    self.embedding = concatenate([token_encoder_embedding, pos_encoder_embedding])
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))
