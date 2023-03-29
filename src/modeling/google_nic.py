import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Add, Dropout, Embedding, Reshape, Bidirectional, Input


class GoogleNIC(Model):

    def __init__(self, vocab_size, embedding_dim) -> None:
        super(GoogleNIC, self).__init__(name='decoder')

        self.dropoutA = Dropout(0.5)
        self.denseA =  Dense(200, activation='relu')

        self.EmbeddingB = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.dropoutB = Dropout(0.5)
       
        self.add = Add()

        self.LSTM = Bidirectional(LSTM(256))
        self.dense2 = Dense(256, activation='relu')
        self.dense3 = Dense(vocab_size, activation='softmax')
    
    def call(self, inputs):
        
        InputA = inputs[0]
        InputB = inputs[1]

        A = self.dropoutA(InputA)
        A = self.denseA(A)

        B = self.EmbeddingB(InputB)
        B = self.dropoutB(B)

        decoder = self.add([A, B])
        decoder = self.LSTM(decoder)
        decoder = self.dense2(decoder)
        outputs = self.dense3(decoder)

        return outputs