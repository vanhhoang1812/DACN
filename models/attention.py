from keras.models import Model
from keras.layers import Input, LSTM, Dense, Attention, Concatenate
from .base import BaseChatbotModel
import numpy as np

class LSTMAttentionChatbotModel(BaseChatbotModel):
    def __init__(self, word2idx, idx2word, embedding_matrix, max_len, latent_dim, model_weights_path):
        super().__init__(word2idx, idx2word, embedding_matrix, max_len, latent_dim)
        self._build_model()
        self.model.load_weights(model_weights_path)
        self._build_inference_models()

    def _build_model(self):
        encoder_inputs = Input(shape=(self.MAX_LEN,))
        encoder_embedding = self.embedding_layer(encoder_inputs)
        encoder_lstm = LSTM(self.LATENT_DIM, return_sequences=True, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

        decoder_inputs = Input(shape=(self.MAX_LEN,))
        decoder_embedding = self.embedding_layer(decoder_inputs)
        decoder_lstm = LSTM(self.LATENT_DIM, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

        attention = Attention()
        attention_output = attention([decoder_outputs, encoder_outputs])
        decoder_concat = Concatenate(axis=-1)([decoder_outputs, attention_output])

        decoder_dense = Dense(self.VOCAB_SIZE, activation='softmax')
        decoder_outputs = decoder_dense(decoder_concat)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.encoder_outputs = encoder_outputs
        self.encoder_lstm = encoder_lstm
        self.decoder_lstm = decoder_lstm
        self.attention = attention
        self.decoder_dense = decoder_dense

    def _build_inference_models(self):
        decoder_state_input_h = Input(shape=(self.LATENT_DIM,))
        decoder_state_input_c = Input(shape=(self.LATENT_DIM,))
        decoder_hidden_state_input = Input(shape=(self.MAX_LEN, self.LATENT_DIM))

        dec_emb2 = self.embedding_layer(self.decoder_inputs)
        decoder_outputs2, h2, c2 = self.decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])
        attention_output_inf = self.attention([decoder_outputs2, decoder_hidden_state_input])
        decoder_concat_input = Concatenate(axis=-1)([decoder_outputs2, attention_output_inf])
        decoder_outputs2 = self.decoder_dense(decoder_concat_input)

        self.decoder_model = Model(
            [self.decoder_inputs, decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
            [decoder_outputs2, h2, c2]
        )

    def decode_sequence(self, input_seq):
        enc_outs, h, c = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.word2idx["<SOS>"]

        decoded_sentence = []
        for _ in range(self.MAX_LEN):
            output_tokens, h, c = self.decoder_model.predict([target_seq, enc_outs, h, c])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.idx2word.get(sampled_token_index, "<UNK>")

            if sampled_word == "<EOS>":
                break

            decoded_sentence.append(sampled_word)
            target_seq[0, 0] = sampled_token_index

        return ' '.join(decoded_sentence)
