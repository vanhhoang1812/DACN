import pickle
import numpy as np

from models.lstm import LSTMChatbotModel
from models.lstm_beam import LSTMBeamSearchChatbotModel
from models.bilstm import BiLSTMChatbotModel
from models.bilstm_beam import BiLSTMBeamSearchChatbotModel
from models.attention import LSTMAttentionChatbotModel

class ModelFactory:
    def __init__(self, tokenizer_path, embedding_path):
        with open(tokenizer_path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.vocab = data['vocab']
            self.MAX_LEN = data['MAX_LEN']

        self.VOCAB_SIZE = len(self.vocab)
        self.EMBEDDING_DIM = 300
        self.embedding_matrix = self._load_embedding_matrix(embedding_path)

    def _load_embedding_matrix(self, embedding_path):
        embedding_index = {}
        with open(embedding_path, encoding='utf-8') as f:
            for line in f:
                values = line.rstrip().split(' ')
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = vector

        embedding_matrix = np.zeros((self.VOCAB_SIZE, self.EMBEDDING_DIM))
        for word, idx in self.word2idx.items():
            embedding_matrix[idx] = embedding_index.get(word, np.random.normal(scale=0.6, size=(self.EMBEDDING_DIM,)))
        return embedding_matrix

    def get_model(self, model_name):
        model_name = model_name.lower()

        if model_name == 'lstm':
            return LSTMChatbotModel(self.word2idx, self.idx2word, self.embedding_matrix, self.MAX_LEN, 256, './model/my_model_lstm.h5')

        elif model_name == 'lstm_beam':
            return LSTMBeamSearchChatbotModel(self.word2idx, self.idx2word, self.embedding_matrix, self.MAX_LEN, 256, './model/my_model_lstm.h5')

        elif model_name == 'bilstm':
            return BiLSTMChatbotModel(self.word2idx, self.idx2word, self.embedding_matrix, self.MAX_LEN, 128, './model/my_model_bi.h5')

        elif model_name == 'bilstm_beam':
            return BiLSTMBeamSearchChatbotModel(self.word2idx, self.idx2word, self.embedding_matrix, self.MAX_LEN, 128, './model/my_model_bi.h5')

        elif model_name == 'lstm_attention':
            return LSTMAttentionChatbotModel(self.word2idx, self.idx2word, self.embedding_matrix, self.MAX_LEN, 256, './model/model_attention.h5')

        else:
            raise ValueError(f"Không hỗ trợ mô hình '{model_name}'")
