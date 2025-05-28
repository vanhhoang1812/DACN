from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate
from .base import BaseChatbotModel
import numpy as np

class BiLSTMChatbotModel(BaseChatbotModel):
    def __init__(self, word2idx, idx2word, embedding_matrix, max_len, latent_dim, model_weights_path):
        super().__init__(word2idx, idx2word, embedding_matrix, max_len, latent_dim)
        self.LATENT_DIM_BI = latent_dim
        self.LATENT_DIM_BI_2 = latent_dim * 2
        self._build_model()
        self.seq2seq_model.load_weights(model_weights_path)
        self._build_inference_models()

    def _build_model(self):
        encoder_inputs = Input(shape=(self.MAX_LEN,))
        encoder_embedding = self.embedding_layer(encoder_inputs)
        encoder_bi_lstm = Bidirectional(LSTM(self.LATENT_DIM_BI, return_state=True))
        encoder_outputs, f_h, f_c, b_h, b_c = encoder_bi_lstm(encoder_embedding)
        state_h = Concatenate()([f_h, b_h])
        state_c = Concatenate()([f_c, b_c])
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(self.MAX_LEN,))
        decoder_embedding = self.embedding_layer(decoder_inputs)
        decoder_lstm = LSTM(self.LATENT_DIM_BI_2, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = Dense(self.VOCAB_SIZE, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.seq2seq_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.encoder_states = encoder_states
        self.decoder_lstm = decoder_lstm
        self.decoder_dense = decoder_dense

    def _build_inference_models(self):
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)

        decoder_state_input_h = Input(shape=(self.LATENT_DIM_BI_2,))
        decoder_state_input_c = Input(shape=(self.LATENT_DIM_BI_2,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_embedding_inf = self.embedding_layer(self.decoder_inputs)
        decoder_outputs_inf, h, c = self.decoder_lstm(decoder_embedding_inf, initial_state=decoder_states_inputs)
        decoder_outputs_inf = self.decoder_dense(decoder_outputs_inf)

        self.decoder_model = Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [decoder_outputs_inf, h, c]
        )

    def decode_sequence(self, input_seq):
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.word2idx["<SOS>"]

        decoded_sentence = []
        for _ in range(self.MAX_LEN):
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.idx2word[sampled_token_index]

            if sampled_word in ("<EOS>", "<PAD>"):
                break

            decoded_sentence.append(sampled_word)
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]

        return ' '.join(decoded_sentence)
