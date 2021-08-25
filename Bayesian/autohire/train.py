import re
from collections import defaultdict

import numpy as np

from .hyperparams import *


def clean_text(text):
    text = text.lower()
    text = re.sub("[^a-z]", " ", text)
    data = text.split()
    data = list(filter(lambda x: len(x) >= WORD_LENGTH_THRESHOLD, data))
    return data

def encode_labels(labels):
    label_values = labels.unique()
    word_to_idx = {word: idx for idx, word in enumerate(label_values)}
    return labels.apply(lambda x: word_to_idx[x])

def generate_dictionary(df):
    dictionary = defaultdict(lambda: np.zeros(max(df["Tag"]) + 1))
    for desc, tag in zip(df["Keywords"], df["Tag"]):
        for word in set(desc):
            dictionary[word][tag] += 1

    words, counts = [], []
    for word, count in dictionary.items():
        if np.sum(count) > WORD_COUNT_THRESHOLD:
            words.append(word)
            counts.append(count)
    return np.stack(counts), np.array(words)
