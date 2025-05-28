from keras.models import Model
from keras.layers import Input, LSTM, Dense
from .base import BaseChatbotModel
import numpy as np

class LSTMChatbotModel(BaseChatbotModel):
    def __init__(self, word2idx, idx2word, embedding_matrix, max_len, latent_dim, model_weights_path):
        super().__init__(word2idx, idx2word, embedding_matrix, max_len, latent_dim)
        self._build_model()
        self.seq2seq_model.load_weights(model_weights_path)
        self._build_inference_models()

    def _build_model(self):
        # Encoder
        encoder_inputs = Input(shape=(self.MAX_LEN,))
        encoder_embedding = self.embedding_layer(encoder_inputs)
        encoder_lstm = LSTM(self.LATENT_DIM, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(self.MAX_LEN,))
        decoder_embedding = self.embedding_layer(decoder_inputs)
        decoder_lstm = LSTM(self.LATENT_DIM, return_sequences=True, return_state=True)
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

        decoder_state_input_h = Input(shape=(self.LATENT_DIM,))
        decoder_state_input_c = Input(shape=(self.LATENT_DIM,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_single_input = Input(shape=(1,))
        decoder_single_embed = self.embedding_layer(decoder_single_input)
        decoder_outputs2, state_h2, state_c2 = self.decoder_lstm(decoder_single_embed, initial_state=decoder_states_inputs)
        decoder_outputs2 = self.decoder_dense(decoder_outputs2)

        self.decoder_model = Model(
            [decoder_single_input] + decoder_states_inputs,
            [decoder_outputs2, state_h2, state_c2]
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

            if sampled_word == "<EOS>":
                break

            decoded_sentence.append(sampled_word)
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]

        return ' '.join(decoded_sentence)