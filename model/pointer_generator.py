from tensorflow.keras.layers import Dense, Activation, multiply, concatenate, Reshape, Flatten
from tensorflow.keras import backend as K


SWITCH_OUTPUT_SIZE = 3


class PointerGenerator:
    def __init__(self, input_size, switch_input_size, vocab_size):
        self.linear = Dense(vocab_size)
        self.softmax = Activation('softmax')
        self.linear_pointer = Dense(SWITCH_OUTPUT_SIZE)
        self.vocab_size = vocab_size
        self.input_size = input_size
    
    def run(self, hiddens, source_attentions, source_attention_maps, target_attentions, target_attention_maps):

        print("HIDDENS: ",K.int_shape(hiddens))
        print("SOURCE ATTENTIONS: ",K.int_shape(source_attentions))
        print("TARGET ATTENTIONS: ",K.int_shape(target_attentions))
        batch_size, num_target_nodes, hidden_size = K.int_shape(hiddens)
        source_dynamic_vocab_size = source_attention_maps.shape[2]
        target_dynamic_vocab_size = target_attention_maps.shape[2]
        hiddens = Flatten()(hiddens)
        
        # Pointer probability.
        p = self.linear_pointer(hiddens)
        p_copy_source = p[:, 0]
        p_copy_target = p[:, 1]
        p_generate = p[:, 2]

        # Probability distribution over the vocabulary.
        scores = self.linear(hiddens)
        scores = Reshape((num_target_nodes, hidden_size))(scores)
        
        vocab_probs = self.softmax(scores)
        expanded_p_generate = K.expand_dims(p_generate,1)
        expanded_p_generate = Reshape((vocab_probs.shape[1], vocab_probs.shape[2]))(expanded_p_generate)
        scaled_vocab_probs = multiply([vocab_probs, expanded_p_generate])
        
        expanded_p_copy_source = K.expand_dims(p_copy_source,1)
        expanded_p_copy_source = Reshape((source_attentions.shape[1], source_attentions.shape[2]))(expanded_p_copy_source)
        scaled_source_attentions = multiply([source_attentions, expanded_p_copy_source])
        scaled_copy_source_probs = K.dot(scaled_source_attentions, source_attention_maps)

        expanded_p_copy_target = K.expand_dims(p_copy_target,1)
        expanded_p_copy_target = Reshape((target_attentions.shape[1], target_attentions.shape[2]))(expanded_p_copy_target)
        scaled_target_attentions = multiply([target_attentions, expanded_p_copy_target])
        scaled_copy_target_probs = K.dot(scaled_target_attentions, target_attention_maps)
        
        probs = concatenate([
            scaled_vocab_probs,
            scaled_copy_source_probs,
            scaled_copy_target_probs
        ], axis=2)
        
        predictions = K.max(probs, 2)

        return probs, predictions, source_dynamic_vocab_size, target_dynamic_vocab_size



    


    