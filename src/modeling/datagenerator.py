import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


class DataGenerator(tf.keras.utils.Sequence):
     
    def __init__(self,
                 images: list,
                 features: dict,
                 image_sequences: dict,
                 vocab_size: int, 
                 max_len: int,
                 image_per_batch: int=3, 
                 shuffle: bool=True) -> None:
        """
        Initialize the DataGenerator class.
        :param images: list of images
        :param features: dictionary of image features
        :param image_sequences: dictionary of image sequences
        :param vocab_size: vocabulary size
        :param max_len: maximum length of sequence
        :param image_per_batch: number of images per batch
        :param shuffle: whether to shuffle the data
        """
        super().__init__()
        self.images = images
        self.features = features
        self.image_sequences = image_sequences
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.image_per_batch = image_per_batch
        self.shuffle = shuffle
        self.indexes = np.arange(len(images))
        self.on_epoch_end()


    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, indexes):
        """
        Generates data containing batch_size samples.
        """
        X1, X2, y = [], [], []
        for idx in indexes:
            image_features = self.features[self.images[idx]]
            for seq in self.image_sequences[self.images[idx]]:
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_len)[0]
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]

                    X1.append(image_features)
                    X2.append(in_seq)
                    y.append(out_seq)

        X1 = np.array(X1)
        X2 = np.array(X2)
        y = np.array(y)

        return X1, X2, y


    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.images) / self.image_per_batch)) 


    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        indexes = self.indexes[index*self.image_per_batch:(index+1)*self.image_per_batch]
        X1, X2, y = self.__data_generation(indexes)
        return [X1, X2], y