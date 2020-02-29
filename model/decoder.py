import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import concatenate, Embedding, LSTM, Lambda, Reshape
from .attention import BahdanauAttention
from .glove_embedding import GloveEmbedding


class Decoder(tf.keras.Model):
    NUM_DECODER_TOKENS = 28
    DECODER_LATENT_DIM = 400
    OUTPUT_DIM = 100

    def __init__(self, token_vocab_size, pos_vocab_size, embedding_dim, dec_units, batch_sz):

        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.token_embedding = GloveEmbedding(
            token_vocab_size, Decoder.NUM_DECODER_TOKENS)
        self.pos_embedding = Embedding(
            pos_vocab_size, output_dim=Decoder.OUTPUT_DIM)
        self.lstm = LSTM(
                Decoder.DECODER_LATENT_DIM,
                return_state=True)
        self.fc = tf.keras.layers.Dense(token_vocab_size)

        # used for attention
        self.source_attention = BahdanauAttention(100)
        self.coref_attention = BahdanauAttention(100)

    def call(self, token_input, pos_input, hidden, enc_output):
        print("enc_output: ", enc_output.shape)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        token_decoder_embedding = self.token_embedding(token_input)
        pos_decoder_embedding = self.pos_embedding(pos_input)
        dec_input = concatenate([token_decoder_embedding, pos_decoder_embedding])
        print("X: ", dec_input.shape)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)

        outputs = []
        states = []
        source_copy_attentions = []
        target_copy_hidden_states = []
        target_copy_attentions = []

        for timestep in range(Decoder.NUM_DECODER_TOKENS):
            current_word = Lambda(lambda x: x[:, timestep: timestep+1, :])(dec_input)
            # x = tf.concat([expanded_context_vector, current_word], axis=-1)
            x = current_word
            # passing the concatenated vector to the LSTM
            output, state_h, state_c  = self.lstm(x, initial_state=hidden)
            hidden_state = concatenate([state_h, state_c])
            output, attention_weight = self.source_attention(output, enc_output)
            # expanded_context_vector = tf.expand_dims(context_vector, 1)
            
            if len(target_copy_hidden_states) == 0:
                target_copy_attention = np.zeros((self.batch_sz, dec_input.shape[1], 1))
            else:
                target_copy_memory = tf.stack(target_copy_hidden_states, 1)
                _, target_copy_attention = self.coref_attention(
                    output, target_copy_memory)
                target_copy_attention = tf.pad(
                            target_copy_attention, tf.constant([[0,0], [0,dec_input.shape[1]-timestep], [0, 0]])
                        )   
 
            outputs.append(output)
            target_copy_hidden_states.append(output)
            states.append(hidden_state)
            source_copy_attentions.append(attention_weight)
            target_copy_attentions.append(target_copy_attention)
        
        x = tf.stack(outputs, axis=1)
        states = tf.stack(states, axis=0)
        source_copy_attentions = tf.squeeze(tf.stack(source_copy_attentions, axis=1), axis=-1)
        target_copy_attentions = tf.squeeze(tf.stack(target_copy_attentions, axis=1), axis=-1)
        print("X: ", x.shape)
        print("STATES: ", x.shape)
        print("source_copy_attentions: ", source_copy_attentions.shape)
        print("target_copy_attentions: ", target_copy_attentions.shape)

        return x, states, source_copy_attentions, target_copy_attentions
