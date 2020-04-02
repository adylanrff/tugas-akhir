import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, multiply, concatenate, Reshape, Flatten, TimeDistributed, Lambda
from tensorflow.keras.activations import softmax
from tensorflow.keras import backend as K
from .util.utils import create_generator_indices

SWITCH_OUTPUT_SIZE = 3


class PointerGenerator(tf.keras.Model):
    def __init__(self, vocab_size):
        super(PointerGenerator, self).__init__()
        linear_pointer = Dense(SWITCH_OUTPUT_SIZE)
        linear = Dense(vocab_size)
        self.softmax = softmax
        self.linear = linear
        self.linear_pointer = linear_pointer
        self.vocab_size = vocab_size
        self.eps = 1e-15
        self.vocab_pad_idx = 0

    def call(self, hiddens, source_attentions, source_attention_maps, target_attentions, target_attention_maps, invalid_indexes=None):

        batch_size, num_target_nodes, hidden_size = K.int_shape(hiddens)
        source_dynamic_vocab_size = source_attention_maps.shape[2]
        target_dynamic_vocab_size = target_attention_maps.shape[2]

        # Pointer probability.
        hiddens = tf.reshape(hiddens, (batch_size*num_target_nodes, -1))
        p = self.softmax(self.linear_pointer(hiddens), axis=1)
        p_copy_source = tf.reshape(tf.expand_dims(p[:, 0], axis=1), (batch_size, num_target_nodes, 1))
        p_copy_target = tf.reshape(tf.expand_dims(p[:, 1], axis=1), (batch_size, num_target_nodes, 1))
        p_generate = tf.reshape(tf.expand_dims(p[:, 2], axis=1), (batch_size, num_target_nodes, 1))

        # Probability distribution over the vocabulary.
        scores = self.linear(hiddens).numpy()
        scores[:, self.vocab_pad_idx] = -float('inf')
        scores = tf.reshape(scores, shape=(batch_size, num_target_nodes, -1))
        vocab_probs = self.softmax(scores)
        # [batch_size, num_target_nodes, vocab_size]
        scaled_vocab_probs = tf.cast(vocab_probs, dtype='float32') * tf.cast(tf.broadcast_to(p_generate, vocab_probs.shape), dtype='float32')

        # [batch_size, num_target_nodes, num_source_nodes]
        scaled_source_attentions = tf.cast(source_attentions, dtype='float32') *  tf.cast(tf.broadcast_to(p_copy_source, source_attentions.shape), dtype='float32')
        # [batch_size, num_target_nodes, dynamic_vocab_size]
        scaled_copy_source_probs = tf.matmul(
            scaled_source_attentions, tf.cast(source_attention_maps, dtype='float32'))

        # [batch_size, num_target_nodes, dynamic_vocab_size]
        scaled_target_attentions = tf.cast(target_attentions, dtype='float32') * tf.broadcast_to(p_copy_target, target_attentions.shape)
        # [batch_size, num_target_nodes, dymanic_vocab_size]
        scaled_copy_target_probs = tf.matmul(
            scaled_target_attentions, tf.cast(target_attention_maps, dtype='float32'))

        if invalid_indexes:
            if invalid_indexes.get('vocab', None) is not None:
                vocab_invalid_indexes = invalid_indexes['vocab']
                vocab_invalid_indexes_mask = np.ones(shape=scaled_vocab_probs.shape)
                for i, indexes in enumerate(vocab_invalid_indexes):
                    for index in indexes:
                        vocab_invalid_indexes_mask[i, :, index] = 0
                scaled_vocab_probs = scaled_vocab_probs * vocab_invalid_indexes_mask
                
            if invalid_indexes.get('source_copy', None) is not None:
                source_copy_invalid_indexes = invalid_indexes['source_copy']
                source_copy_invalid_indexes_mask = np.ones(shape=scaled_copy_source_probs.shape)
                for i, indexes in enumerate(source_copy_invalid_indexes):
                    for index in indexes:
                        source_copy_invalid_indexes_mask[i, :, index] = 0
                scaled_copy_source_probs = scaled_copy_source_probs * source_copy_invalid_indexes_mask

        probs = concatenate([
            scaled_vocab_probs,
            scaled_copy_source_probs,
            scaled_copy_target_probs
        ], axis=2)

        predictions = K.argmax(probs, 2)
        return probs, predictions

    def compute_loss(self, probs, predictions, generate_targets,
                     source_copy_targets, source_dynamic_vocab_size,
                     target_copy_targets, target_dynamic_vocab_size, copy_attentions):
        """
        Priority: target_copy > source_copy > generate

        :param probs: probability distribution,
            [batch_size, num_target_nodes, vocab_size + dynamic_vocab_size]
        :param predictions: [batch_size, num_target_nodes]
        :param generate_targets: target node index in the vocabulary,
            [batch_size, num_target_nodes]
        :param source_copy_targets:  target node index in the dynamic vocabulary,
            [batch_size, num_target_nodes]
        :param source_dynamic_vocab_size: int
        :param target_copy_targets:  target node index in the dynamic vocabulary,
            [batch_size, num_target_nodes]
        :param target_dynamic_vocab_size: int
        :param coverage_records: None or a tensor recording source-side coverages.
            [batch_size, num_target_nodes, num_source_nodes]
        :param copy_attentions: [batch_size, num_target_nodes, num_source_nodes]
        """
        non_pad_mask = tf.cast(tf.math.not_equal(generate_targets, self.vocab_pad_idx), dtype='int32')

        source_copy_mask = tf.math.not_equal(source_copy_targets,
            1) & tf.math.not_equal(source_copy_targets,0)  # 1 is the index for unknown words
        source_copy_mask = tf.cast(source_copy_mask, dtype='int32')
        non_source_copy_mask = tf.convert_to_tensor(1 - source_copy_mask, dtype='int32')

        target_copy_mask = tf.cast(tf.math.not_equal(target_copy_targets, 0), dtype='int32')  # 0 is the index for coref NA
        non_target_copy_mask = tf.convert_to_tensor(1 - target_copy_mask, dtype='int32')

        # [batch_size, num_target_nodes, 1]
        target_copy_targets_with_offset = tf.expand_dims(target_copy_targets, 2) + self.vocab_size + source_dynamic_vocab_size
        target_copy_targets_indices = create_generator_indices(tf.squeeze(target_copy_targets_with_offset))

        # [batch_size, num_target_nodes]
        target_copy_target_probs = tf.gather_nd(probs, target_copy_targets_indices)
        target_copy_target_probs = target_copy_target_probs * tf.cast(target_copy_mask, dtype='float32')

        # [batch_size, num_target_nodes, 1]
        source_copy_targets_with_offset = tf.expand_dims(source_copy_targets, 2) + self.vocab_size
        source_copy_targets_indices = create_generator_indices(tf.squeeze(source_copy_targets_with_offset))

        # [batch_size, num_target_nodes]
        source_copy_target_probs = tf.gather_nd(probs, source_copy_targets_indices)
        
        source_copy_target_probs = source_copy_target_probs * tf.cast(non_target_copy_mask, dtype='float32') * tf.cast(source_copy_mask, dtype='float32')

        # [batch_size, num_target_nodes]
        generate_targets_indices = create_generator_indices(generate_targets)
        generate_target_probs = tf.gather_nd(probs, generate_targets_indices)

        # Except copy-oov nodes, all other nodes should be copied.
        mul_result = generate_target_probs * tf.cast(non_target_copy_mask, dtype='float32') * tf.cast(non_source_copy_mask, dtype='float32')
        likelihood = target_copy_target_probs + source_copy_target_probs + mul_result
        num_tokens = tf.math.reduce_sum(non_pad_mask)

        # Add eps for numerical stability.
        likelihood = likelihood + self.eps

        # Drop pads.
        loss = -tf.math.log(likelihood) * tf.cast(non_pad_mask, dtype='float32')

        # Mask out copy targets for which copy does not happen.
        targets = tf.squeeze(target_copy_targets_with_offset) * target_copy_mask + \
            tf.squeeze(source_copy_targets_with_offset) * non_target_copy_mask * source_copy_mask + \
            generate_targets * non_target_copy_mask * non_source_copy_mask
            
        targets = targets * non_pad_mask

        pred_eq = tf.cast(tf.math.equal(predictions, tf.cast(targets, dtype='int64')), dtype='int32')
        pred_eq =  pred_eq * non_pad_mask

        num_non_pad = tf.math.reduce_sum(non_pad_mask)
        num_correct_pred = tf.math.reduce_sum(pred_eq)

        num_target_copy = tf.math.reduce_sum(target_copy_mask * non_pad_mask)
        num_correct_target_copy = tf.math.reduce_sum(pred_eq * target_copy_mask)
        
        num_correct_target_point = tf.cast(tf.math.greater_equal(predictions, (self.vocab_size + source_dynamic_vocab_size)), dtype='int32')
        num_correct_target_point = num_correct_target_point * target_copy_mask * non_pad_mask
        num_correct_target_point = tf.math.reduce_sum(num_correct_target_point)

        num_source_copy = tf.math.reduce_sum(source_copy_mask * non_target_copy_mask * non_pad_mask)

        num_correct_source_copy = tf.math.reduce_sum(pred_eq * non_target_copy_mask * source_copy_mask)

        num_correct_source_point = tf.cast(tf.math.greater_equal(predictions, self.vocab_size), dtype='int32') * \
            tf.cast(tf.math.less(predictions, (self.vocab_size + source_dynamic_vocab_size)), dtype='int32') * \
            non_target_copy_mask * source_copy_mask * non_pad_mask

        num_correct_source_point = tf.math.reduce_sum(num_correct_source_point)

        return dict(
            loss=tf.math.reduce_sum(loss) / tf.cast(num_tokens, dtype='float32'),
            total_loss=tf.math.reduce_sum(loss),
            num_tokens=num_tokens,
            predictions=predictions
        )
