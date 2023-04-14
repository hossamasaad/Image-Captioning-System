import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Add, Dropout, Embedding, Reshape, Bidirectional


class BahdanauAttention(Model):

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        score = self.V(attention_hidden_layer)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class NICAttention(tf.keras.Model):

    def __init__(self, embedding_dim, vocab_size):
        super(NICAttention, self).__init__()
        self.dropout1 = Dropout(0.3)
        self.dense1 = Dense(embedding_dim, activation='relu')

        self.embeddingB = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.dropoutB = Dropout(0.5)
        self.LSTM = Bidirectional(LSTM(256, return_sequences=True, return_state=True))

        self.attention = BahdanauAttention(256)

        self.dense2 = Dense(256, activation='relu')
        self.dense3 = Dense(vocab_size, activation='softmax')
        self.hidden = tf.zeros((3, 256))
    
    def call(self, inputs):
        InputA = inputs[0]
        InputB = inputs[1]

        A = self.dropout1(InputA)
        A = self.dense1(A)

        context_vector, attention_weights = self.attention(A, self.hidden)
        
        B = self.embeddingB(InputB)
        
        x = tf.concat([tf.expand_dims(context_vector, 1), B], axis=-1)

        output, state = self.LSTM(x)
        self.hidden = state

        x = self.dense2(output)
        x = self.dense3(x) 

        return x