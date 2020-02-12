import torch
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional, concatenate, Flatten
from .glove_embedding import GloveEmbedding

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
        encoder_output, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(
            TextToAMR.ENCODER_LATENT_DIM,
            return_sequences=True, 
            return_state=True))(encoder_embedding)
        state_h = concatenate([forward_h, backward_h])
        state_c = concatenate([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_outputs, _, _ = LSTM(
            TextToAMR.DECODER_LATENT_DIM,
            return_sequences=True, 
            return_state=True)(decoder_embedding, initial_state=encoder_states)
        
        flattened = Flatten()(decoder_outputs)
        temp_output = Dense(1)(flattened)
        
        # TODO: Add Source Attention layer
        source_attention = []
        # TODO: Add Coref Attention layer
        coref_attention = []
        # TODO: Add Pointer generator model
        pointer_generator = []

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