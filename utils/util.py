from constants import *

import json
import numpy as np
import subprocess
import fastText
import msgpack

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
            path_to_train=KNOWN_EMBEDDING_PATH,
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
            path_to_emb=KNOWN_EMBEDDING_PATH,
            path_to_i2w=INDEX2WORD_PATH
    ):
        #self.__train_embeddings = np.load(path_to_emb)
        with open(F_EMBEDDING_PATH, 'rb') as f:
            self.__train_embeddings = np.array(
                    msgpack.load(f, encoding='utf8')['embedding'],
                    dtype=np.float32
            )
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
        return self.__train_embeddings[key]
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
def load_train(path=F_COMFORT_TRAIN_PATH):
    data = np.load(path)
    print("____________")
    print(data['context_int_features'].shape)
    print("____________")
    return (
            (data['context'], data['context_len'], data['context_int_features'], data['context_float_features']),
            (data['question'], data['question_len']),
            (data['answer_begin'], data['answer_end'])
           )
# load_train

def load_test(path=F_COMFORT_TEST_PATH):
    data = np.load(path)
    return (
            (data['context'], data['context_len'], data['context_features']),
            (data['question'], data['question_len']),
            (data['answer_begin'], data['answer_end'])
           )
# load_train

def get_random_batch(data, batch_size, embedding):
    n = data[0][0].shape[0]
    indexes = np.random.choice(n, batch_size, replace=False)
    context = embedding.get_known(data[0][0][indexes])
    context_len = data[0][1][indexes]
    context_int_features = data[0][2][indexes]
    context_float_features = data[0][3][indexes]
    question = embedding.get_known(data[1][0][indexes])
    question_len = data[1][1][indexes]
    answer_begin = data[2][0][indexes]
    answer_end = data[2][1][indexes]
    return (
            (context, context_len, context_int_features, context_float_features),
            (question, question_len),
            (answer_begin, answer_end)
        )
# get_random_batch

def get_batch(data, l, r, embedding):
    context = embedding.get_known(data[0][0][l:r])
    context_len = data[0][1][l:r]
    context_int_features = data[0][2][l:r]
    context_float_features = data[0][3][l:r]
    question = embedding.get_known(data[1][0][l:r])
    question_len = data[1][1][l:r]
    answer_begin = data[2][0][l:r]
    answer_end = data[2][1][l:r]
    return (
            (context, context_len, context_int_features, context_float_features),
            (question, question_len),
            (answer_begin, answer_end)
        )
# get_batch

def shuffle(data):
    shuffle = np.random.permutation(data[0][0].shape[0])
    context = data[0][0][shuffle]
    context_len = data[0][1][shuffle]
    context_int_features = data[0][2][shuffle]
    context_float_features = data[0][3][shuffle]
    question = data[1][0][shuffle]
    question_len = data[1][1][shuffle]
    answer_begin = data[2][0][shuffle]
    answer_end = data[2][1][shuffle]
    return (
            (context, context_len, context_int_features, context_float_features),
            (question, question_len),
            (answer_begin, answer_end)
           )
# shuffle

def next_batch(data, batch_size, embedding):
    data = shuffle(data)
    for l in range(0, data[0][0].shape[0], batch_size):
        r = min(l + batch_size, data[0][0].shape[0])
        yield get_batch(data, l, r, embedding)
# next_batch

def get_answer_mask(max_len, window):
    mask = np.zeros((max_len, max_len), dtype=np.float32)
    for i in range(max_len):
        mask[i, i+1:min(i+window+1, max_len)] = 1
    return mask
# get_answer_mask

def find_answer(exp_begin, exp_end, window):
    batch_size = exp_begin.shape[0]
    size = exp_begin.shape[1]
    answer_begin = np.empty((batch_size,), dtype=np.int32)
    answer_end = np.empty((batch_size,), dtype=np.int32)
    for i in range(batch_size):
        max_point = 0
        for b in range(size - 1):
            for e in range(b + 1, min(b + window + 1, size)):
                point = exp_begin[i, b] * exp_end[i, e]
                if point > max_point:
                    max_point = point
                    answer_begin[i] = b
                    answer_end[i] = e
    return answer_begin, answer_end
# find_answer

