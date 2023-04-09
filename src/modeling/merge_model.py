from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Add, Dropout, Embedding, Reshape, Bidirectional


class MergeDecoder(Model):

    def __init__(self, vocab_size, embedding_dim) -> None:
        super(MergeDecoder, self).__init__(name='decoder')

        self.dropout1 = Dropout(0.5)
        self.dense1 =  Dense(512, activation='relu')

        self.Embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.dropout2 = Dropout(0.5)
        self.LSTM = Bidirectional(LSTM(256))

        self.add = Add()
        self.dense2 = Dense(256, activation='relu')

        self.dense3 = Dense(vocab_size, activation='softmax')
    
    def call(self, inputs):
        
        InputA = inputs[0]
        InputB = inputs[1]

        A = self.dropout1(InputA)
        A = self.dense1(A)

        B = self.Embedding(InputB)
        B = self.dropout2(B)
        B = self.LSTM(B)

        decoder = self.add([A, B])
        decoder = self.dense2(decoder)

        outputs = self.dense3(decoder)

        return outputs