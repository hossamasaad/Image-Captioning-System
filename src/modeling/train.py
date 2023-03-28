import os
import pickle
import argparse
import numpy as np
from merge_model import MergeDecoder
from datagenerator import DataGenerator
from tensorflow.keras.layers import Input

MODELS = {
    "merge": MergeDecoder(vocab_size=8433, embbeding_dim=200, max_length=201)
}


def load(path):
    """
    Load pickle files
    Args:
        path (str): path of the pickle file 
    Returns:
        results (object): loaded objec
    """
    with open(path, "rb") as file:
        result = pickle.load(file)
    return result


def load_embedding_matrix(embedding_dim, vocab_size, word_index):
    """"""
    embeddings_index = {} 
    f = open('glove.6B.200d.txt', encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix        


def save_model(model, save_path):
    """
    Save model
    Args:
        model (keras.Model): model to save
        save_path (str): path to save model
    """
    with open(save_path) as file:
        pickle.dump(model, file)


def train(model, generator, epochs, embedding_matrix):

    # Inputs
    inputs1 = Input(shape=(2048,))
    inputs2 = Input(shape=(201,))

    # build model
    model([inputs1, inputs2])

    # Add embedding matrix
    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False

    # Compile model
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam"
    )

    # start training
    model.fit(generator, epochs=epochs)

    return model

def main(args):
    # Model
    model = MODELS[args.model]
    epochs = args.epochs

    # Load data
    images = os.listdir(args.images)
    image_features = load(args.images_features)
    image_sequences = load(args.images_sequences)
    
    word_index = load(args.word_index)
    embedding_matrix = load_embedding_matrix(embedding_dim=200, vocab_size=8433, word_index=word_index)

    # create generator
    generator = DataGenerator(
        images=images,
        features=image_features,
        image_sequences=image_sequences,
        vocab_size=8433,
        max_len=201,
    )

    # start training
    model = train(
        model=model,
        generator=generator,
        epochs=epochs,
        embedding_matrix=embedding_matrix
    )

    # save model
    save_model(model, save_path=f"{args.model_name}_v1.pkl")


if __name__ == "__main__":

    # parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--images_features", type=str)
    parser.add_argument("--images_sequences", type=str)
    parser.add_argument("--images", type=str)
    parser.add_argument("--word_index", type=str)

    main(parser.parse_args())

