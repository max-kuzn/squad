import json
from tqdm import tqdm
import numpy as np

from constants import *

def make_vocabulary(vocabulary, tokenized_data):
    for par in tqdm(tokenized_data['paragraphs']):
        for token in par['context']:
            vocabulary.add(token)
        for qa in par['qas']:
            for token in qa['question']:
                vocabulary.add(token)
    return vocabulary

def main():
    tokenized_train = json.load(open(TOKENIZED_TRAIN_PATH, 'r'))
    tokenized_test = json.load(open(TOKENIZED_TEST_PATH, 'r'))
    print('Read data')
    vocabulary = set()
    vocabulary = make_vocabulary(vocabulary, tokenized_train)
    vocabulary = make_vocabulary(vocabulary, tokenized_test)
    print('Made set')
    index2word = list(vocabulary)
    print('Made list')
    np.save(INDEX2WORD_PATH, index2word)

if __name__ == "__main__":
    main()

