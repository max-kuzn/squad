from constants import *

import numpy as np
import subprocess
import fastText

def tokenize(string):
    tokenizer = subprocess.Popen(
            [TOKENIZER_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
    )
    tokens = tokenizer.communicate(
                string.encode("utf-8")
             )[0].decode("utf-8").split()
    low_tokens = [t.lower() for t in tokens]
    return low_tokens

class Embedding:
    def __init__(
            self,
            mode="train", # Can be "train", "test", "both"
            path_to_train=TRAIN_EMBEDDING_PATH,
            path_to_train_index2word=INDEX2WORD_PATH,
            path_to_bin=ALL_EMBEDDING_PATH
    ):
        self.__emb_size = 300
        self.__mode = mode
        self.__train_embeddings = None # numpy array
        self.__train_index2word = None
        self.__train_word2index = None
        self.__all_embeddings = None   # fastText model
        if mode == "train" or mode == "both":
            self.__load_train_embeddings__(
                    path_to_emb=path_to_train,
                    path_to_i2w=path_to_train_index2word
            )
        if mode == "test" or mode == "both":
            self.__load_all_embeddings__(path=path_to_bin)

    def __load_train_embeddings__(
            self,
            path_to_emb=TRAIN_EMBEDDING_PATH,
            path_to_i2w=INDEX2WORD_PATH
    ):
        self.__train_embeddings = np.load(path_to_emb)
        i2w = np.load(path_to_i2w)
        w2i = dict()
        for i in range(i2w.shape[0]):
            w2i[i2w[i]] = i
        self.__train_index2word = i2w
        self.__train_word2index = w2i

    def __load_all_embeddings__(self, path=ALL_EMBEDDING_PATH):
        self.__all_embeddings = fastText.load_model(path)

    def get_known(self, key):
        if self.__mode == "test":
            raise
        if np.issubdtype(type(key), np.integer):
            return self.__train_embeddings[key]
        elif isinstance(key, list):
            res = list()
            for k in key:
                res.append(self.get_known(k))
            return res
        elif isinstance(key, np.ndarray):
            res = np.empty(key.shape + (self.__emb_size, ),
                    dtype=np.float32)
            for i in range(res.shape[0]):
                res[i] = self.get_known(key[i])
            return res
        else:
            print(type(key))
            raise

