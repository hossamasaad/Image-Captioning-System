import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3


def feature_extractor():
    """
    Create the inception model
    Returns:
        an Inception model to extract features
    """
    inception = InceptionV3(weights='imagenet')

    my_input = inception.input
    my_output = inception.layers[-2].output

    return Model(inputs=my_input, outputs=my_output)


def preprocess(image_path) -> None:
    """
    Preprocess images befor feeding to the encoder  
    Args:
        image_path: path of image to load it
    Returns:
        x: preprocessed image
    """
    
    # load all images and convert to (299, 299) as expected by InceptionV3
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
    
    x = tf.keras.preprocessing.image.img_to_array(img)          # Convert PIL image to numpy array
    x = np.expand_dims(x, axis=0)                               # add one more dimension
    x = tf.keras.applications.inception_v3.preprocess_input(x)  # preprocess the image to be ready for InceptionV3
    return x


def encode(images_path):
    """
    Start encoding images and extract features
    Args:
        image_path: path of image to load it
    
    Returns:
        features: extracted features
    """
    images = os.listdir(images_path)
    assert len(images) == 8091

    # start encoding images
    features = {}
    encoder = feature_extractor()
    for image in tqdm(images):
        # Image path
        image_path = f"{images_path}/{image}"
        # load image to be ready for encoding
        preprocessed_image = preprocess(image_path)
        
        # encoding
        image_features = encoder.predict(preprocessed_image)
        image_features = np.reshape(image_features, image_features.shape[1])

        features[image] = image_features

    return features
    

def save_features(features: dict, save_path: str) -> None:
    """
    Pickle captions map and save it
    Args:
        captions (dict): Captions map to save
        save_path (str): path to save captions
    """
    with open(save_path, "wb") as file:
        pickle.dump(features, file)


if __name__ == "__main__":
    
    # Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str)
    args = parser.parse_args()

    IMAGES_PATH = args.images_path

    # Encode images
    features = encode(images_path=IMAGES_PATH)

    # Save images
    save_features(features=features, save_path="data/features.pkl")