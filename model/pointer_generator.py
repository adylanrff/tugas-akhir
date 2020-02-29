import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, multiply, concatenate, Reshape, Flatten, TimeDistributed, Lambda
from tensorflow.keras.activations import softmax
from tensorflow.keras import backend as K


SWITCH_OUTPUT_SIZE = 3


class PointerGenerator(tf.keras.Model):
    def __init__(self, vocab_size):
        super(PointerGenerator, self).__init__()
        self.linear = Dense(vocab_size)
        self.softmax = softmax
        self.linear_pointer = TimeDistributed(Dense(SWITCH_OUTPUT_SIZE))
        self.vocab_size = vocab_size
    
    def call(self, hiddens, source_attentions, source_attention_maps, target_attentions, target_attention_maps):

        print("HIDDENS: ",K.int_shape(hiddens))
        print("SOURCE ATTENTION MAP: ",K.int_shape(source_attention_maps))
        print("TARGET ATTENTION MAP: ",K.int_shape(target_attention_maps))
        batch_size, num_target_nodes, hidden_size = K.int_shape(hiddens)
        source_dynamic_vocab_size = source_attention_maps.shape[2]
        target_dynamic_vocab_size = target_attention_maps.shape[2]
        # hiddens = Flatten()(hiddens)
        
        # Pointer probability.
        p = self.softmax(self.linear_pointer(hiddens), axis=1)
        print("p: ", p.shape)
        p_copy_source = p[:, :, 0]
        p_copy_target = p[:, :, 1]
        p_generate = p[:, :, 2]
        
        print("p_copy_source: ", p_copy_source.shape)
        print("p_copy_target: ", p_copy_target.shape)
        print("p_generate: ", p_generate.shape)

        # Probability distribution over the vocabulary.
        scores = self.linear(hiddens)
        vocab_probs = self.softmax(scores)
        print("scores: ", vocab_probs.shape)

        # [batch_size, num_target_nodes, vocab_size]
        scaled_vocab_probs = Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=-1))([vocab_probs, p_generate])
        print("scaled_vocab_probs: ", scaled_vocab_probs.shape)

        # [batch_size, num_target_nodes, num_source_nodes]
        scaled_source_attentions = Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=-1))([source_attentions, p_copy_source])
        # [batch_size, num_target_nodes, dynamic_vocab_size]
        scaled_copy_source_probs = tf.matmul(scaled_source_attentions, source_attention_maps)
        print("scaled_copy_source_probs: ", scaled_copy_source_probs.shape)
        
        # [batch_size, num_target_nodes, dynamic_vocab_size]
        scaled_target_attentions = Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=-1))([target_attentions, p_copy_target])
        # [batch_size, num_target_nodes, dymanic_vocab_size]
        scaled_copy_target_probs = tf.matmul(scaled_target_attentions, target_attention_maps)
        print("scaled_copy_target_probs: ", scaled_copy_target_probs.shape)
        
        probs = concatenate([
            scaled_vocab_probs,
            scaled_copy_source_probs,
            scaled_copy_target_probs
        ], axis=2)
        
        predictions = K.max(probs, 2)
        print(predictions.shape)
        
        return probs, predictions



    


    