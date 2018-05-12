from constants import *

import json
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
# tokenize

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
            self.__load_train_embeddings(
                    path_to_emb=path_to_train,
                    path_to_i2w=path_to_train_index2word
            )
        if mode == "test" or mode == "both":
            self.__load_all_embeddings(path=path_to_bin)
    # __init__

    def __load_train_embeddings(
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
    # __load_train_embeddings

    def __load_all_embeddings(self, path=ALL_EMBEDDING_PATH):
        self.__all_embeddings = fastText.load_model(path)
    # __load_all_embeddings

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
    # get_known
# Embedding

# train data in format:
# train = (context_data, question_data, answer_data)
# Where
# context_data  = (context,      context_len)
# question_data = (quiestion,    quiestion_len)
# answer        = (answer_begin, answer_end)
def load_train(path=COMFORT_TRAIN_PATH):
    data = np.load(path)
    return (
            (data['context'], data['context_len']),
            (data['question'], data['question_len']),
            (data['answer_begin'], data['answer_end'])
           )
# load_train

def get_batch(train, batch_size, embedding):
    N = train[0][0].shape[0]
    indexes = np.random.choice(N, batch_size, replace=False)
    context = embedding.get_known(train[0][0][indexes])
    context_len = train[0][1][indexes]
    question = embedding.get_known(train[1][0][indexes])
    question_len = train[1][1][indexes]
    answer_begin = train[2][0][indexes]
    answer_end = train[2][1][indexes]
    return (
            (context, context_len),
            (question, question_len),
            (answer_begin, answer_end)
           )
# get_batch
