import os
import re
import pickle
import argparse


def clean_caption(caption: str) -> str:
    """
    Clear symbols, numbers and single letters from caption
    Args:
        caption (str): uncleared caption
    Returns:
        caption (str): cleared caption
    """
    # removing symbols and numbers
    caption = re.sub(r'[!@#.$(),"%^*?:;~`0-9]', ' ', caption)
    caption = re.sub(r'[[]]', ' ', caption)

    # removing single letters
    caption = ' '.join( [w for w in caption.split() if len(w)>1] )

    # convert to lower case
    caption = caption.lower()

    return caption


def load_captions(captions_path: str) -> dict:
    """
    Load captions from text file and create a map from image to its captions
    Args:
        captions_path (str): text file captions path
    Returns:
        captions (dict): captions map
    """
    captions = {}

    with open(captions_path) as file:
        for line in file.readlines()[1:]:
            # Get image name and its caption
            image, caption = line.strip().replace(",", " ").split(".jpg")
            image += ".jpg"
            
            # Check if the caption in the dictionary or not
            if image not in captions:
                captions[image] = []

            # Clean caption
            caption = clean_caption(caption)

            # Add caption to dictionary
            captions[image].append(caption)

    return captions


def save_captions(captions: dict, save_path: str) -> None:
    """
    Pickle captions map and save it
    Args:
        captions (dict): Captions map to save
        save_path (str): path to save captions
    """
    with open(save_path, "wb") as file:
        pickle.dump(captions, file)


def main():
    # Pasparsing data path
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()
    
    assert type(args.data_path) == str


    # Set Images and Captions paths
    IMAGES_PATH = f"{args.data_path}/Images"
    CAPTIONS_PATH = f"{args.data_path}/captions.txt"
    assert len(os.listdir(IMAGES_PATH)) == 8091

    # Load captions into dataframe
    captions = load_captions(captions_path=CAPTIONS_PATH)
    assert len(captions) == 8091
    
    # save captions
    save_captions(captions, "data/clean_captions.pkl")
    
    return captions


if __name__ == "__main__":
    cap = main()