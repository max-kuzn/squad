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
            path_to_train=F_EMBEDDING_PATH
    ):
        self.__emb_size = 300
        self.__mode = mode
        self.__train_embeddings = None # numpy array
        if mode == "train" or mode == "both":
            self.__load_train_embeddings(
                    path_to_emb=path_to_train
            )
    # __init__

    def __load_train_embeddings(
            self,
            path_to_emb
    ):
        with open(path_to_emb, 'rb') as f:
            self.__train_embeddings = np.array(
                    msgpack.load(f, encoding='utf8')['embedding'],
                    dtype=np.float32
            )
    # __load_train_embeddings

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
    return (
            (data['context'], data['context_len']),
            (data['context_int_features'], data['context_float_features']),
            (data['question'], data['question_len']),
            (data['answer_begin'], data['answer_end'])
           )
# load_train

def load_test(path=F_COMFORT_TEST_PATH):
    data = np.load(path)
    return (
            (data['context'], data['context_len']),
            (data['context_int_features'], data['context_float_features']),
            (data['question'], data['question_len']),
            (data['answer_begin'], data['answer_end'])
           )
# load_train

def get_random_batch(data, batch_size, embedding):
    n = data[0][0].shape[0]
    indexes = np.random.choice(n, batch_size, replace=False)
    context = embedding.get_known(data[0][0][indexes])
    context_len = data[0][1][indexes]
    context_int_features = data[1][0][indexes]
    context_float_features = data[1][1][indexes]
    question = embedding.get_known(data[2][0][indexes])
    question_len = data[2][1][indexes]
    answer_begin = data[3][0][indexes]
    answer_end = data[3][1][indexes]
    return (
            (context, context_len),
            (context_int_features, context_float_features),
            (question, question_len),
            (answer_begin, answer_end)
        )
# get_random_batch

def get_batch(data, l, r, embedding):
    context = embedding.get_known(data[0][0][l:r])
    context_len = data[0][1][l:r]
    context_int_features = data[1][0][l:r]
    context_float_features = data[1][1][l:r]
    question = embedding.get_known(data[2][0][l:r])
    question_len = data[2][1][l:r]
    answer_begin = data[3][0][l:r]
    answer_end = data[3][1][l:r]
    return (
            (context, context_len),
            (context_int_features, context_float_features),
            (question, question_len),
            (answer_begin, answer_end)
        )
# get_batch

def shuffle(data):
    shuffle = np.random.permutation(data[0][0].shape[0])
    context = data[0][0][shuffle]
    context_len = data[0][1][shuffle]
    context_int_features = data[1][0][shuffle]
    context_float_features = data[1][1][shuffle]
    question = data[2][0][shuffle]
    question_len = data[2][1][shuffle]
    answer_begin = data[3][0][shuffle]
    answer_end = data[3][1][shuffle]
    return (
            (context, context_len),
            (context_int_features, context_float_features),
            (question, question_len),
            (answer_begin, answer_end),
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

def find_one_answer(prob_begin, prob_end, window):
    max_p = 0
    begin = 0
    end = 0
    for i in range(prob_begin.shape[0]):
        for j in range(i, min(i + window, prob_end.shape[0])):
            p = prob_begin[i] * prob_end[j]
            if p > max_p:
                max_p = p
                begin = i
                end = j
    return begin, end
# find_one_answer

def find_answer(prob_begin, prob_end, window):
    batch_size = prob_begin.shape[0]
    answer_begin = np.empty((batch_size,), dtype=np.int32)
    answer_end = np.empty((batch_size,), dtype=np.int32)
    for i in range(batch_size):
        answer_begin[i], answer_end[i] = find_one_answer(
                prob_begin[i], prob_end[i], window
        )
    return answer_begin, answer_end
# find_answer

def one_f1_score(answer_begin, answer_end, true_answer_begin, true_answer_end):
    if answer_end <= answer_begin:
        return 0
    tp = min(answer_end, true_answer_end) \
            - max(answer_begin, true_answer_begin)
    if tp <= 0:
        return 0
    a = max(answer_end, true_answer_end) \
            - min(answer_begin, true_answer_begin)
    return 2 * tp / (tp + a)
# one_f1_score

def f1_score(answer_begin, answer_end, true_answer_begin, true_answer_end, mode='avg'):
    n = 0
    f1 = 0
    for i in range(answer_begin.shape[0]):
        f1 += one_f1_score(
                answer_begin[i],
                answer_end[i],
                true_answer_begin[i],
                true_answer_end[i]
            )
    if mode == 'avg':
        return f1 / n
    if mode == 'sum':
        return f1
    else:
        return 0
# f1_score

