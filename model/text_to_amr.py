import torch
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional, concatenate, Flatten, Lambda, Reshape, Attention
from .glove_embedding import GloveEmbedding
from .attention import BahdanauAttention
from .pointer_generator import PointerGenerator
from stog.utils.string import START_SYMBOL, END_SYMBOL, find_similar_token, is_abstract_token
from .encoder import Encoder
from .decoder import Decoder

class TextToAMR(Model):
    NUM_ENCODER_TOKENS = 25
    NUM_DECODER_TOKENS = 28
    ENCODER_LATENT_DIM = 200
    DECODER_LATENT_DIM = ENCODER_LATENT_DIM*2
    COPY_ATTENTION_MAPS = 27
    COREF_ATTENTION_MAPS = 29

    def __init__(self, vocab, debug=False):
        super().__init__()
        self.vocab = vocab
        self.encoder_token_vocab_size = self.vocab.get_vocab_size("encoder_token_ids")
        self.decoder_token_vocab_size = self.vocab.get_vocab_size("decoder_token_ids")
        self.encoder_pos_vocab_size = self.decoder_pos_vocab_size = self.vocab.get_vocab_size("pos_tags")
        self.is_prepared_input = False
        print("ENCODER TOKEN VOCAB SIZE: ", self.encoder_token_vocab_size)
        print("DECODER TOKEN VOCAB SIZE: ", self.decoder_token_vocab_size)
        
    def generate_model(self):
        if not self.is_prepared_input:
            raise Exception("Please run .prepare_input() first!")
        # Inputs
        ## Encoder input
        token_encoder_input = Input(shape=(TextToAMR.NUM_ENCODER_TOKENS, ), dtype='int32', name="token_encoder_input", batch_size=40)
        pos_encoder_input = Input(shape=(TextToAMR.NUM_ENCODER_TOKENS, ), dtype='int32',name="pos_encoder_input", batch_size=40)
        ## Decoder input
        token_decoder_input = Input(shape=(TextToAMR.NUM_DECODER_TOKENS, ), dtype='int32', name="token_decoder_input", batch_size=40)
        pos_decoder_input = Input(shape=(TextToAMR.NUM_DECODER_TOKENS, ), dtype='int32',name="pos_decoder_input", batch_size=40)
        
        ## Generator input
        copy_attention_maps_input = Input(shape=(TextToAMR.NUM_ENCODER_TOKENS, TextToAMR.COPY_ATTENTION_MAPS, ),dtype='float32',name="copy_attention_maps_input", batch_size=40)
        coref_attention_maps_input = Input(shape=(TextToAMR.NUM_DECODER_TOKENS, TextToAMR.COREF_ATTENTION_MAPS, ),dtype='float32',name="coref_attention_maps_input", batch_size=40)

        # Encoder-Decoder
        encoder = Encoder(self.encoder_token_vocab_size, self.encoder_pos_vocab_size, 100, TextToAMR.ENCODER_LATENT_DIM, 40)
        decoder = Decoder(self.decoder_token_vocab_size, self.decoder_pos_vocab_size, 100, TextToAMR.DECODER_LATENT_DIM, 40)
        
        # Pointer Generator
        pointer_generator = PointerGenerator(self.decoder_token_vocab_size)

        # Training

        sample_hidden = encoder.initialize_hidden_state()
        enc_output, enc_hidden = encoder(token_encoder_input, pos_encoder_input, sample_hidden)
        dec_output, dec_hidden, source_copy_attentions, target_copy_attentions = decoder(token_decoder_input, pos_decoder_input, enc_hidden, enc_output)
        
        print("DECODER_OUTPUT: ", dec_output.shape)
        print("DECODER_HIDDEN: ", dec_hidden.shape)
        print("Source copy Attentions: ", source_copy_attentions.shape)
        print("Target copy attentions: ", target_copy_attentions.shape)

        # # Pass to pointer generator
        probs, predictions = pointer_generator(
            dec_output, 
            source_copy_attentions, 
            copy_attention_maps_input,
            target_copy_attentions,
            coref_attention_maps_input
            )

        source_dynamic_vocab_size, target_dynamic_vocab_size = copy_attention_maps_input.shape[2], coref_attention_maps_input.shape[2]
        
        flattened = Flatten()(predictions)
        temp_output = Dense(1)(flattened)
        
        # TODO: Add Deep Biaffine Decoder model
        biaffine_decoder = []

        return Model([
            token_encoder_input, 
            pos_encoder_input, 
            token_decoder_input, 
            pos_decoder_input, 
            copy_attention_maps_input, 
            coref_attention_maps_input
        ], temp_output) 


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
            bert_token=bert_token_inputs.numpy() if bert_token_inputs is not None else None,
            token_subword_index=encoder_token_subword_index.numpy() if encoder_token_subword_index is not None else None,
            token=encoder_token_inputs.numpy() if encoder_token_inputs is not None else None,
            pos_tag=encoder_pos_tags.numpy() if encoder_pos_tags is not None else None,
            must_copy_tag=encoder_must_copy_tags.numpy() if encoder_must_copy_tags is not None else None,
            char=encoder_char_inputs.numpy() if encoder_char_inputs is not None else None,
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
            token=decoder_token_inputs.numpy() if decoder_token_inputs is not None else None,
            pos_tag=decoder_pos_tags.numpy() if decoder_pos_tags is not None else None,
            char=decoder_char_inputs.numpy() if decoder_char_inputs is not None else None,
            coref=decoder_coref_inputs.numpy() if decoder_coref_inputs is not None else None
        )

        # [batch, num_tokens]
        vocab_targets = data['tgt_tokens']['decoder_tokens'][:, 1:].contiguous()
        # [data, num_tokens]
        coref_targets = data["tgt_copy_indices"][:, 1:]
        # [data, num_tokens, num_tokens + coref_na]
        coref_attention_maps = data['tgt_copy_map'][:, 1:]  # exclude BOS
        # [data, num_tgt_tokens, num_src_tokens + unk]
        copy_targets = data["src_copy_indices"][:, 1:]
        # [data, num_src_tokens + unk, src_dynamic_vocab_size]
        # Exclude the last pad.
        copy_attention_maps = data['src_copy_map'][:, 1:-1]

        generator_inputs = dict(
            vocab_targets=vocab_targets.numpy() if vocab_targets is not None else None,
            coref_targets=coref_targets.numpy() if coref_targets is not None else None,
            coref_attention_maps=coref_attention_maps.numpy() if coref_attention_maps is not None else None,
            copy_targets=copy_targets.numpy() if copy_targets is not None else None,
            copy_attention_maps=copy_attention_maps.numpy() if copy_attention_maps is not None else None
        )

        # Remove the last two pads so that they have the same size of other inputs?
        edge_heads = data['head_indices'][:, :-2]
        edge_labels = data['head_tags'][:, :-2]
        # TODO: The following computation can be done in amr.py.
        # Get the parser mask.
        parser_token_inputs = torch.zeros_like(decoder_token_inputs)
        parser_token_inputs.copy_(decoder_token_inputs)
        parser_token_inputs[
            parser_token_inputs == self.vocab.get_token_index(END_SYMBOL, 'decoder_token_ids')
        ] = 0
        parser_mask = (parser_token_inputs != 0).float()

        parser_inputs = dict(
            edge_heads=edge_heads.numpy() if edge_heads is not None else None,
            edge_labels=edge_labels.numpy() if edge_labels is not None else None,
            corefs=decoder_coref_inputs.numpy() if decoder_coref_inputs is not None else None,
            mask=parser_mask.numpy() if parser_mask is not None else None
        )
        
        self.is_prepared_input = True

        print("ENCODER_INPUT")
        self.__print_tensor_dim(encoder_inputs)
        
        print("DECODER_INPUT")
        self.__print_tensor_dim(decoder_inputs)

        print("GENERATOR_INPUT")
        self.__print_tensor_dim(generator_inputs)

        print("PARSER_INPUT")
        self.__print_tensor_dim(parser_inputs)

        return (encoder_inputs, decoder_inputs, generator_inputs, parser_inputs)

    def __print_tensor_dim(self, data):
        for key in data:
            print(key, end= ": ")
            if data[key] is not None:
                print(data[key].shape)
            else:
                print("None")
        print("")