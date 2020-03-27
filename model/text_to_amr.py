import torch
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional, concatenate, Flatten, Lambda, Reshape, Attention
from stog.utils.string import START_SYMBOL, END_SYMBOL, find_similar_token, is_abstract_token
from stog.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN

from .glove_embedding import GloveEmbedding
from .attention import BahdanauAttention
from .pointer_generator import PointerGenerator
from .biaffine_decoder import DeepBiaffineDecoder
from .encoder import Encoder
from .decoder import Decoder
from .util import get_text_field_mask

class TextToAMR():
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
        self.max_decode_length = TextToAMR.NUM_DECODER_TOKENS

        punctuation_ids = []
        oov_id = vocab.get_token_index(DEFAULT_OOV_TOKEN, 'decoder_token_ids')
        for c in ',.?!:;"\'-(){}[]':
            c_id = vocab.get_token_index(c, 'decoder_token_ids')
            if c_id != oov_id:
                punctuation_ids.append(c_id)
        self.punctuation_ids = punctuation_ids
        
        # Encoder-Decoder
        self.encoder = Encoder(self.encoder_token_vocab_size, self.encoder_pos_vocab_size, 100, TextToAMR.ENCODER_LATENT_DIM, 40)
        self.decoder = Decoder(self.decoder_token_vocab_size, self.decoder_pos_vocab_size, 100, TextToAMR.DECODER_LATENT_DIM, 40)
        
        # Pointer Generator
        self.pointer_generator = PointerGenerator(self.decoder_token_vocab_size)

        # Deep Biaffine Decoder
        self.biaffine_decoder = DeepBiaffineDecoder(self.vocab)

        self.models = [self.encoder, self.decoder, self.pointer_generator, self.biaffine_decoder]

        self.optimizer = tf.keras.optimizers.Adam()

    def train(self, model_input): 
        if not self.is_prepared_input:
            raise Exception("Please run .prepare_input() first!")
        #########################################
        ## TRAINING                             #
        #########################################
        loss = 0
        with tf.GradientTape() as tape:
            token_encoder_input, pos_encoder_input, token_decoder_input, pos_decoder_input, copy_attention_maps_input, coref_attention_maps_input, parser_mask_input, edge_heads_input, edge_labels_input, corefs_input, vocab_target_input, coref_target_input, copy_target_input = model_input
            sample_hidden = self.encoder.initialize_hidden_state()
            enc_output, enc_hidden = self.encoder(token_encoder_input, pos_encoder_input, sample_hidden)
            dec_output, dec_hidden, rnn_hidden_states, source_copy_attentions, target_copy_attentions = self.decoder(token_decoder_input, pos_decoder_input, enc_hidden, enc_output)

            # # Pass to pointer generator
            # output: [probs, predictions, source_dynamic_vocab_size, target_dynamic_vocab_size]
            probs, predictions = self.pointer_generator(
                dec_output, 
                source_copy_attentions, 
                copy_attention_maps_input,
                target_copy_attentions,
                coref_attention_maps_input
                )

            source_dynamic_vocab_size, target_dynamic_vocab_size = copy_attention_maps_input.shape[2], coref_attention_maps_input.shape[2]
            generator_loss = self.pointer_generator.compute_loss(
                probs, 
                predictions, 
                vocab_target_input,
                copy_target_input,
                source_dynamic_vocab_size,
                coref_target_input,
                target_dynamic_vocab_size,
                source_copy_attentions
                )
            
            # pass to biaffine decoder
            # output: [edge_heads, edge_labels, loss, total_loss, num_nodes]
            edge_heads_output, edge_labels_output, biaffine_decoder_loss, biaffine_decoder_total_loss, biaffine_decoder_num_nodes = self.biaffine_decoder(
                rnn_hidden_states, 
                edge_heads_input, 
                edge_labels_input, 
                corefs_input, 
                parser_mask_input
            )

            loss += generator_loss['loss'] + biaffine_decoder_loss

        variables = sum([model.trainable_variables for model in self.models], [])
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss, generator_loss['loss'], biaffine_decoder_loss

    def predict(self, model_input):
        token_encoder_input, pos_encoder_input, token_decoder_input, pos_decoder_input, copy_attention_maps_input, coref_attention_maps_input, parser_mask_input, edge_heads_input, edge_labels_input, corefs_input, vocab_target_input, coref_target_input, copy_target_input, mask_encoder_input, src_copy_vocab,tag_lut,source_copy_invalid_ids   = model_input
        sample_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(token_encoder_input, pos_encoder_input, sample_hidden)
        
        invalid_indexes = dict(
            source_copy= source_copy_invalid_ids,
            vocab=[set(self.punctuation_ids) for _ in range(len(tag_lut))]
        )

        encoder_memory_bank=enc_output
        encoder_mask=mask_encoder_input,
        encoder_final_states=enc_hidden,
        copy_attention_maps=copy_attention_maps_input,
        copy_vocabs=src_copy_vocab,
        tag_luts=tag_lut,
        invalid_indexes=invalid_indexes

        generator_outputs = self.decode_with_pointer_generator(
                encoder_memory_bank, mask_encoder_input, encoder_final_states, copy_attention_maps, copy_vocabs, tag_luts, invalid_indexes)
    
        parser_outputs = self.decode_with_graph_parser(
            generator_outputs['decoder_inputs'],
            generator_outputs['decoder_rnn_memory_bank'],
            generator_outputs['coref_indexes'],
            generator_outputs['decoder_mask']
        )

        return dict(
            nodes=generator_outputs['predictions'],
            heads=parser_outputs['edge_heads'],
            head_labels=parser_outputs['edge_labels'],
            corefs=generator_outputs['coref_indexes'],
        )

    def decode_with_pointer_generator(
            self, memory_bank, mask, states, copy_attention_maps, copy_vocabs,
            tag_luts, invalid_indexes):
        # [batch_size, 1]
        batch_size = memory_bank.size(0)

        tokens = tf.ones(shape=(batch_size, 1)) * self.vocab.get_token_index(
            START_SYMBOL, "decoder_token_ids")
        pos_tags = tf.ones(shape=(batch_size, 1)) * self.vocab.get_token_index(
            DEFAULT_OOV_TOKEN, "pos_tags")
        tokens = tokens
        corefs = tf.zeros(shape=(batch_size, 1))

        decoder_input_history = []
        decoder_outputs = []
        rnn_outputs = []
        copy_attentions = []
        coref_attentions = []
        predictions = []
        coref_indexes = []
        decoder_mask = []

        input_feed = None
        coref_inputs = []

        # A sparse indicator matrix mapping each node to its index in the dynamic vocab.
        # Here the maximum size of the dynamic vocab is just max_decode_length.
        coref_attention_maps = tf.zeros(
            shape=(batch_size, self.max_decode_length, self.max_decode_length + 1))
        # A matrix D where the element D_{ij} is for instance i the real vocab index of
        # the generated node at the decoding step `i'.
        coref_vocab_maps = tf.zeros(shape=(batch_size, self.max_decode_length + 1))

        coverage = None

        for step_i in range(self.max_decode_length):
            # 1. Get the decoder inputs.
            # token_embeddings = self.decoder_token_embedding(tokens)
            # pos_tag_embeddings = self.decoder_pos_embedding(pos_tags)
            # coref_embeddings = self.decoder_coref_embedding(corefs)
            
            # decoder_inputs = torch.cat(
            #     [token_embeddings, pos_tag_embeddings, coref_embeddings], 2)
            # decoder_inputs = self.decoder_embedding_dropout(decoder_inputs)

            # 2. Decode one step.
            dec_output, dec_hidden, rnn_hidden_states, source_copy_attentions, target_copy_attentions, states = self.decoder(token_decoder_input, pos_decoder_input, states, memory_bank)

            # 3. Run pointer/generator.
            if step_i == 0:
                _coref_attention_maps = coref_attention_maps[:, :step_i + 1]
            else:
                _coref_attention_maps = coref_attention_maps[:, :step_i]

            generator_output = self.generator(
                _decoder_outputs, _copy_attentions, copy_attention_maps,
                _coref_attentions, _coref_attention_maps, invalid_indexes)
            _predictions = generator_output['predictions']

            # 4. Update maps and get the next token input.
            tokens, _predictions, pos_tags, corefs, _mask = self._update_maps_and_get_next_input(
                step_i,
                generator_output['predictions'].squeeze(1),
                generator_output['source_dynamic_vocab_size'],
                coref_attention_maps,
                coref_vocab_maps,
                copy_vocabs,
                decoder_mask,
                tag_luts,
                invalid_indexes
            )

            # 5. Update variables.
            decoder_input_history += [decoder_inputs]
            decoder_outputs += [_decoder_outputs]
            rnn_outputs += [_rnn_outputs]

            copy_attentions += [_copy_attentions]
            coref_attentions += [_coref_attentions]

            predictions += [_predictions]
            # Add the coref info for the next input.
            coref_indexes += [corefs]
            # Add the mask for the next input.
            decoder_mask += [_mask]

        # 6. Do the following chunking for the graph decoding input.
        # Exclude the hidden state for BOS.
        decoder_input_history = torch.cat(decoder_input_history[1:], dim=1)
        decoder_outputs = torch.cat(decoder_outputs[1:], dim=1)
        rnn_outputs = torch.cat(rnn_outputs[1:], dim=1)
        # Exclude coref/mask for EOS.
        # TODO: Answer "What if the last one is not EOS?"
        predictions = torch.cat(predictions[:-1], dim=1)
        coref_indexes = torch.cat(coref_indexes[:-1], dim=1)
        decoder_mask = 1 - torch.cat(decoder_mask[:-1], dim=1)

        return dict(
            # [batch_size, max_decode_length]
            predictions=predictions,
            coref_indexes=coref_indexes,
            decoder_mask=decoder_mask,
            # [batch_size, max_decode_length, hidden_size]
            decoder_inputs=decoder_input_history,
            decoder_memory_bank=decoder_outputs,
            decoder_rnn_memory_bank=rnn_outputs,
            # [batch_size, max_decode_length, encoder_length]
            copy_attentions=copy_attentions,
            coref_attentions=coref_attentions
        )

    def decode_with_graph_parser(self, decoder_inputs, memory_bank, corefs, mask):
        """Predict edges and edge labels between nodes.
        :param decoder_inputs: [batch_size, node_length, embedding_size]
        :param memory_bank: [batch_size, node_length, hidden_size]
        :param corefs: [batch_size, node_length]
        :param mask:  [batch_size, node_length]
        :return a dict of edge_heads and edge_labels.
            edge_heads: [batch_size, node_length]
            edge_labels: [batch_size, node_length]
        """
        if self.use_aux_encoder:
            aux_encoder_outputs = self.aux_encoder(decoder_inputs, mask)
            self.aux_encoder.reset_states()
            memory_bank = torch.cat([memory_bank, aux_encoder_outputs], 2)

        memory_bank, _, _, corefs, mask = self.graph_decoder._add_head_sentinel(
            memory_bank, None, None, corefs, mask)
        (edge_node_h, edge_node_m), (edge_label_h, edge_label_m) = self.graph_decoder.encode(memory_bank)
        edge_node_scores = self.graph_decoder._get_edge_node_scores(edge_node_h, edge_node_m, mask.float())
        edge_heads, edge_labels = self.graph_decoder.mst_decode(
            edge_label_h, edge_label_m, edge_node_scores, corefs, mask)
        return dict(
            edge_heads=edge_heads,
            edge_labels=edge_labels
        )

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
        encoder_mask = get_text_field_mask(data['src_tokens'])

        encoder_inputs = dict(
            bert_token=bert_token_inputs.numpy() if bert_token_inputs is not None else None,
            token_subword_index=encoder_token_subword_index.numpy() if encoder_token_subword_index is not None else None,
            token=encoder_token_inputs.numpy() if encoder_token_inputs is not None else None,
            pos_tag=encoder_pos_tags.numpy() if encoder_pos_tags is not None else None,
            must_copy_tag=encoder_must_copy_tags.numpy() if encoder_must_copy_tags is not None else None,
            char=encoder_char_inputs.numpy() if encoder_char_inputs is not None else None,
            mask=encoder_mask
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
        edge_heads = data['head_indices'][:, :-1]
        edge_labels = data['head_tags'][:, :-1]
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