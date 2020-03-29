import tensorflow as tf

def get_text_field_mask(text_field_tensors,
                        num_wrapping_dims=0):
    """
    Takes the dictionary of tensors produced by a ``TextField`` and returns a mask
    with 0 where the tokens are padding, and 1 otherwise.  We also handle ``TextFields``
    wrapped by an arbitrary number of ``ListFields``, where the number of wrapping ``ListFields``
    is given by ``num_wrapping_dims``.
    If ``num_wrapping_dims == 0``, the returned mask has shape ``(batch_size, num_tokens)``.
    If ``num_wrapping_dims > 0`` then the returned mask has ``num_wrapping_dims`` extra
    dimensions, so the shape will be ``(batch_size, ..., num_tokens)``.
    There could be several entries in the tensor dictionary with different shapes (e.g., one for
    word ids, one for character ids).  In order to get a token mask, we use the tensor in
    the dictionary with the lowest number of dimensions.  After subtracting ``num_wrapping_dims``,
    if this tensor has two dimensions we assume it has shape ``(batch_size, ..., num_tokens)``,
    and use it for the mask.  If instead it has three dimensions, we assume it has shape
    ``(batch_size, ..., num_tokens, num_features)``, and sum over the last dimension to produce
    the mask.  Most frequently this will be a character id tensor, but it could also be a
    featurized representation of each token, etc.
    If the input ``text_field_tensors`` contains the "mask" key, this is returned instead of inferring the mask.
    TODO(joelgrus): can we change this?
    NOTE: Our functions for generating masks create torch.LongTensors, because using
    torch.ByteTensors  makes it easy to run into overflow errors
    when doing mask manipulation, such as summing to get the lengths of sequences - see below.
    >>> mask = torch.ones([260]).byte()
    >>> mask.sum() # equals 260.
    >>> var_mask = torch.autograd.V(mask)
    >>> var_mask.sum() # equals 4, due to 8 bit precision - the sum overflows.
    """
    if "mask" in text_field_tensors:
        return text_field_tensors["mask"]

    tensor_dims = [(tensor.dim(), tensor)
                   for tensor in text_field_tensors.values()]
    tensor_dims.sort(key=lambda x: x[0])

    smallest_dim = tensor_dims[0][0] - num_wrapping_dims
    if smallest_dim == 2:
        token_tensor = tensor_dims[0][1]
        return (token_tensor != 0).long()
    elif smallest_dim == 3:
        character_tensor = tensor_dims[0][1]
        return ((character_tensor > 0).long().sum(dim=-1) > 0).long()
    else:
        raise ValueError(
            "Expected a tensor with dimension 2 or 3, found {}".format(smallest_dim))

def create_indices(label):
    batch_size = label.shape[0]
    label_length = label.shape[1]
    batch_index = tf.range(batch_size)
    tile = tf.transpose(tf.reshape(tf.tile(batch_index, [label_length]), shape=[label_length, batch_size]))
    tile = tf.split(tile, batch_size, 0)
    splitted_label = tf.split(label, batch_size, 0)
    indices = []
    for idx,head in enumerate(splitted_label):
        matrix_idx = tf.cast(tf.transpose(tile[idx]), dtype='int32')
        head = tf.cast(tf.transpose(head), dtype='int32')    
        indices.append(tf.reshape(tf.stack([matrix_idx,head], axis=-1), shape=[label_length,2]))
    return tf.stack(indices)

def create_edge_node_index(batch_index, edge_heads, modifier_index):   
    batch_index = tf.broadcast_to(tf.expand_dims(batch_index, -1), shape=(edge_heads.shape[0], edge_heads.shape[1]))             
    modifier_index = tf.broadcast_to(tf.transpose(tf.expand_dims(modifier_index, -1)), shape=(edge_heads.shape[0], edge_heads.shape[1]))             
    return tf.stack([batch_index, edge_heads, modifier_index], axis=-1)

def create_edge_label_index(batch_index, modifier_index, edge_labels):
    batch_index = tf.broadcast_to(tf.expand_dims(batch_index, -1), shape=(edge_labels.shape[0], edge_labels.shape[1]))             
    modifier_index = tf.broadcast_to(tf.transpose(tf.expand_dims(modifier_index, -1)), shape=(edge_labels.shape[0], edge_labels.shape[1]))
    return tf.stack([batch_index, modifier_index, edge_labels], axis=-1)

def create_generator_indices(target_copy_targets):
    batch_index = tf.range(target_copy_targets.shape[0])
    batch_index = tf.broadcast_to(tf.expand_dims(batch_index, -1), shape=(target_copy_targets.shape[0], target_copy_targets.shape[1]))
    modifier_index = tf.broadcast_to(tf.transpose(tf.expand_dims(tf.range(target_copy_targets.shape[1]), -1)), shape=(target_copy_targets.shape[0], target_copy_targets.shape[1]))  
    return tf.stack([batch_index, modifier_index, target_copy_targets], axis=-1)

def create_coref_attention_maps_index(coref_index):
    batch_index = tf.range(target_copy_targets.shape[0])
    batch_index = tf.broadcast_to(tf.expand_dims(batch_index, -1), shape=(coref_index.shape[0], coref_index.shape[1]))

def create_coref_maps_index(coref_predictions):
    batch_index = tf.range(coref_predictions.shape[0], dtype='int32')
    return tf.stack([batch_index, tf.cast(coref_predictions, dtype='int32')], axis=-1)
