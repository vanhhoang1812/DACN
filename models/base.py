import numpy as np
from keras.preprocessing.sequence import pad_sequences
from underthesea import word_tokenize

class BaseChatbotModel:
    def __init__(self, word2idx, idx2word, embedding_matrix, max_len, latent_dim):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.embedding_matrix = embedding_matrix
        self.MAX_LEN = max_len
        self.LATENT_DIM = latent_dim
        self.VOCAB_SIZE = embedding_matrix.shape[0]
        self._build_embedding_layer()

    def _build_embedding_layer(self):
        from keras.layers import Embedding
        self.embedding_layer = Embedding(
            input_dim=self.VOCAB_SIZE,
            output_dim=self.embedding_matrix.shape[1],
            weights=[self.embedding_matrix],
            input_length=self.MAX_LEN,
            trainable=False
        )

    def tokenize_and_format(self, sentence):
        tokens = word_tokenize(sentence, format="text").split()
        return ["<SOS>"] + tokens + ["<EOS>"]

    def sentence_to_indices(self, tokens):
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

    def prepare_input(self, sentence):
        tokens = self.tokenize_and_format(sentence)
        seq = self.sentence_to_indices(tokens)
        return pad_sequences([seq], maxlen=self.MAX_LEN, padding='post', value=self.word2idx["<PAD>"])

    def decode_sequence(self, input_seq):
        raise NotImplementedError("Lớp con phải định nghĩa decode_sequence")

    def predict(self, sentence):
        input_seq = self.prepare_input(sentence)
        return self.decode_sequence(input_seq)
