import torch
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional, concatenate, Flatten, Lambda, Reshape
from .glove_embedding import GloveEmbedding
from .attention import Attention

class TextToAMR(Model):
    NUM_ENCODER_TOKENS = 25
    NUM_DECODER_TOKENS = 28
    ENCODER_LATENT_DIM = 200
    DECODER_LATENT_DIM = ENCODER_LATENT_DIM*2

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.encoder_token_vocab_size = self.vocab.get_vocab_size("encoder_token_ids")
        self.decoder_token_vocab_size = self.vocab.get_vocab_size("decoder_token_ids")
        self.encoder_pos_vocab_size = self.decoder_pos_vocab_size = self.vocab.get_vocab_size("pos_tags")
        self.model = self.__generate_model_v1()

    def __generate_model_v1(self):
        # Inputs
        token_encoder_input = Input(shape=(TextToAMR.NUM_ENCODER_TOKENS, ), dtype='int32', name="token_encoder_input")
        pos_encoder_input = Input(shape=(TextToAMR.NUM_ENCODER_TOKENS, ), dtype='int32',name="pos_encoder_input")
        token_decoder_input = Input(shape=(TextToAMR.NUM_DECODER_TOKENS, ), dtype='int32', name="token_decoder_input")
        pos_decoder_input = Input(shape=(TextToAMR.NUM_DECODER_TOKENS, ), dtype='int32',name="pos_decoder_input")
        
        # Embedding
        ## Encoder embedding
        token_encoder_embedding = GloveEmbedding(self.encoder_token_vocab_size, TextToAMR.NUM_ENCODER_TOKENS)(token_encoder_input)
        pos_encoder_embedding = Embedding(input_dim=self.encoder_pos_vocab_size,output_dim=100, input_length=TextToAMR.NUM_ENCODER_TOKENS)(pos_encoder_input)
        encoder_embedding = concatenate([token_encoder_embedding, pos_encoder_embedding])
        
        ## Decoder embedding
        token_decoder_embedding = GloveEmbedding(self.decoder_token_vocab_size, TextToAMR.NUM_DECODER_TOKENS)(token_decoder_input)
        pos_decoder_embedding = Embedding(input_dim=self.decoder_pos_vocab_size,output_dim=100, input_length=TextToAMR.NUM_DECODER_TOKENS)(pos_decoder_input)
        decoder_embedding = concatenate([token_decoder_embedding, pos_decoder_embedding])

        # Encoder
        encoder = Bidirectional(LSTM(
            TextToAMR.ENCODER_LATENT_DIM,
            return_sequences=True, 
            return_state=True))

        # Decoder
        decoder = LSTM(
            TextToAMR.DECODER_LATENT_DIM, 
            return_state=True)
        
        # Source attention
        source_attention = Attention(alignment_type='global', context='many-to-many', name="source_attention")

        # Coref attention 
        coref_attention = Attention(alignment_type='global', context='many-to-many', name="coref_attention")

        # Build model
        
        # Encode
        encoder_output, forward_h, forward_c, backward_h, backward_c = encoder(encoder_embedding)
        state_h = concatenate([forward_h, backward_h])
        state_c = concatenate([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        # Decoder initial states
        rnn_hidden_states = []
        decoder_hidden_states = []
        source_copy_attentions = []
        target_copy_attentions = []

        current_decoder_hidden_state = encoder_states
        memory_bank = encoder_output
    
        target_copy_hidden_states = []

        # Decoder
        for timestep in range(decoder_embedding.shape[1]):
            # decode per timestep
            current_word = Lambda(lambda x: x[timestep: timestep+1, :, :])(decoder_embedding)

            decoder_output, state_h, state_c = decoder(current_word, initial_state=current_decoder_hidden_state)
            current_decoder_hidden_state = [state_h, state_c]
            output = Reshape((1, decoder_output.shape[1]))(decoder_output)
            
            output, source_attention_weights = source_attention([encoder_output, decoder_output, timestep])
            input_feed = output
        
            if (len(target_copy_hidden_states) == 0):
                target_copy_attention = np.zeros(shape=(1, current_word.shape[1], current_word.shape[2]))
            else:
                if (len(target_copy_hidden_states) > 1):
                    target_copy_memory = concatenate(target_copy_hidden_states)
                else:
                    target_copy_memory = target_copy_hidden_states

                _, coref_attention_weights = coref_attention([output, decoder_output, timestep])

            target_copy_attentions.append(target_copy_attention)
            target_copy_hidden_states.append(output)
            decoder_hidden_states.append(output)
        
        flattened = Flatten()(output)
        temp_output = Dense(1)(flattened)
        
        # TODO: Add Pointer generator model
        pointer_generator = []

        # TODO: Add Deep Biaffine Decoder model
        biaffine_decoder = []

        return Model([token_encoder_input, pos_encoder_input, token_decoder_input, pos_decoder_input], temp_output) 


    def prepare_input(self, data):
        # Encoder
        bert_token_inputs = data.get('src_token_ids', None)
        if bert_token_inputs is not None:
            bert_token_inputs = bert_token_inputs.long()
        encoder_token_subword_index = data.get('src_token_subword_index', None)
        if encoder_token_subword_index is not None:
            encoder_token_subword_index = encoder_token_subword_index.long()
        encoder_token_inputs = data['src_tokens']['encoder_tokens']
        encoder_pos_tags = data['src_pos_tags']
        encoder_must_copy_tags = data['src_must_copy_tags']
        # [data, num_tokens, num_chars]
        encoder_char_inputs = data['src_tokens']['encoder_characters']
        # [data, num_tokens]
        # encoder_mask = get_text_field_mask(data['src_tokens'])

        encoder_inputs = dict(
            bert_token=bert_token_inputs,
            token_subword_index=encoder_token_subword_index,
            token=encoder_token_inputs,
            pos_tag=encoder_pos_tags,
            must_copy_tag=encoder_must_copy_tags,
            char=encoder_char_inputs,
            # mask=encoder_mask
        )

        # Decoder
        decoder_token_inputs = data['tgt_tokens']['decoder_tokens'][:, :-1].contiguous()
        decoder_pos_tags = data['tgt_pos_tags'][:, :-1]
        # [data, num_tokens, num_chars]
        decoder_char_inputs = data['tgt_tokens']['decoder_characters'][:, :-1].contiguous()
        # TODO: The following change can be done in amr.py.
        # Initially, raw_coref_inputs has value like [0, 0, 0, 1, 0]
        # where '0' indicates that the input token has no precedent, and
        # '1' indicates that the input token's first precedent is at position '1'.
        # Here, we change it to [0, 1, 2, 1, 4] which means if the input token
        # has no precedent, then it is referred to itself.
        raw_coref_inputs = data["tgt_copy_indices"][:, :-1].contiguous()
        coref_happen_mask = raw_coref_inputs.ne(0)
        decoder_coref_inputs = torch.ones_like(raw_coref_inputs) * torch.arange(
            0, raw_coref_inputs.size(1)).type_as(raw_coref_inputs).unsqueeze(0)
        decoder_coref_inputs.masked_fill_(coref_happen_mask, 0)
        # [data, num_tokens]
        decoder_coref_inputs = decoder_coref_inputs + raw_coref_inputs

        decoder_inputs = dict(
            token=decoder_token_inputs,
            pos_tag=decoder_pos_tags,
            char=decoder_char_inputs,
            coref=decoder_coref_inputs
        )

        return (encoder_inputs, decoder_inputs)