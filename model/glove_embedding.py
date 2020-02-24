from tensorflow.keras.layers import Embedding

class GloveEmbedding(Embedding):
    EMBEDDING_DIM = 100
    
    @classmethod
    def get_pretrained_embedding(cls):
        return []
        
    # TODO: get pretrained embedding matrix

    def __init__(self, vocab_size, input_length):
        super().__init__(
            vocab_size,
            GloveEmbedding.EMBEDDING_DIM,
            input_length=input_length,
            trainable=False
        )
    