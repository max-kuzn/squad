import json
from tqdm import tqdm
import numpy as np

from constants import *

def make_vocabulary(tokenized_data):
    vocabulary = set()
    for par in tqdm(tokenized_data['paragraphs']):
        for token in par['context']:
            vocabulary.add(token)
        for qa in par['qas']:
            for token in qa['question']:
                vocabulary.add(token)
    return vocabulary

def main():
    tokenized_data = json.load(open(TOKENIZED_TRAIN_PATH, 'r'))
    vocabulary = make_vocabulary(tokenized_data)
    print(vocabulary)
    index2word = list(vocabulary)
    np.save(INDEX2WORD_PATH, index2word)

if __name__ == "__main__":
    main()

