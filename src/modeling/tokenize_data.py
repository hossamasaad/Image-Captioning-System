import pickle
import argparse
from tensorflow.keras.preprocessing.text import Tokenizer


def load_captions(captions_path: str) -> dict:
    """
    Load cleaned captions
    Args:
        captions_path (str): cleaned captions path to laod
    Returns:
        captions (dict): loaded captions map
    """
    with open(captions_path, "rb") as file:
        captions = pickle.load(file)
    
    return captions


def add_tokens(captions):
    """
    Add <start> and <end> tokens to captions
    Args:
        Captions: dict of captions before adding tokens
    Returns
        Captions: dict of captions after adding tokens
    """
    for image in captions.keys():
        tmp_captions = []
        for cap in captions[image]:
            cap = '<start> ' + cap + ' <end>'
            tmp_captions.append(cap)
    
        captions[image] = tmp_captions
    
    return captions


def tokenize(captions):
    """
    Tokenize captions
    Args:
        captions (dict): captions map image to captions
    Returns
        captions (dict): caption map image to sequences
    """
    all_captions = []
    for image in captions.keys():
        all_captions += captions[image]
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)

    # Vocablury size
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1

    # Get max length of sequences
    max_len = 0
    for cap in all_captions:
        max_len = max(max_len, len(cap))
    
    print("Size of vocabulary: ", vocab_size)
    print("Max length: ", max_len)

    # tokneize words
    for image in captions.keys():
        captions[image] = tokenizer.texts_to_sequences(captions[image])

    return captions


def save_captions(captions: dict, save_path: str) -> None:
    """
    Pickle captions map and save it
    Args:
        captions (dict): Captions map image_to_sequences to save
        save_path (str): path to save captions
    """
    with open(save_path, "wb") as file:
        pickle.dump(captions, file)


def main():
    
    # Parse caption path
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=str)
    CAPTIONS_PATH = parser.parse_args().captions
    assert CAPTIONS_PATH[-4:] == ".pkl"

    # load captions
    captions = load_captions(captions_path=CAPTIONS_PATH)
    # Add tokens
    captions = add_tokens(captions)
    # tokenize
    captions = tokenize(captions)
    # save image_sequences
    save_captions(captions, "data/image_to_sequences.pkl")

    return captions
    
    
if __name__ == "__main__":
    main()