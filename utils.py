import logging
from string import punctuation, digits
import numpy as np

from typing import List

FILENAME = "./game.tsv"


def load_data(filename: str) -> List[str]:
    try:
        with open(filename) as fi:
            data = fi.readlines()
            data = [i.rstrip() for i in data]
    except FileNotFoundError:
        logging.error(f"File {filename!r} is not found!")
        raise

    return data


def extract_words(line: str) -> List[str]:
    for c in punctuation + digits:
        line.replace(c, f" {c} ")

    return line.lower().split()


def create_bag_of_words(lines: list) -> dict:
    dictionary = {}  # maps word to unique index
    for line in lines:
        word_list = extract_words(line)
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)

    return dictionary


def extract_bow_feature_vector(state_desc: str, dictionary: dict) -> np.ndarray:
    state_vector = np.zeros([len(dictionary)])
    word_list = extract_words(state_desc)
    for word in word_list:
        if word in dictionary:
            state_vector[dictionary[word]] += 1

    return state_vector


def ewma(a: list, alpha: float = 0.9) -> np.ndarray:
    """Computes the exponentially weighted moving average of a"""
    b = np.array(a)
    n = b.size
    w0 = np.ones(n) * alpha
    p = np.arange(n - 1, -1, -1)

    return np.average(b, weights=w0**p)
