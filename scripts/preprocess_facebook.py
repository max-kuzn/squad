import sys
sys.path.insert(0, '../utils')

from constants import *

import numpy as np
from tqdm import tqdm
import msgpack

INT_FEATURES_SIZE = 5
FLOAT_FEATURES_SIZE = 1
MAX_CONTEXT_LEN = 0

def init_comfort_data(data):
    global MAX_CONTEXT_LEN
    max_context_len = 0
    max_question_len = 0
    n = len(data)
    for elem in data:
        max_context_len = max(max_context_len, len(elem[1])
        max_question_len = max(max_question_len, len(elem[5])
    MAX_CONTEXT_LEN = max_context_len
    context = np.zeros((n, max_context_len), dtype=np.int32)
    context_len = np.zeros((n,), dtype=np.int32)
    context_int_features = np.zeros((n, max_context_len, INT_FEATURES_SIZE), dtype=np.int32)
    context_float_features = np.zeros((n, max_context_len, FLOAT_FEATURES_SIZE), dtype=np.float32)
    question = np.zeros((n, max_question_len), dtype=np.int32)
    question_len = np.zeros((n,), dtype=np.int32)
    answer_begin = np.zeros((n,), dtype=np.int32)
    answer_end = np.zeros((n,), dtype=np.int32)
    return (
            (context, context_len, context_int_features, context_float_features),
            (question, question_len),
            (answer_begin, answer_end)
           )
# init_comfort_data

def get_features(elem):
    int_features = np.zeros((MAX_CONTEXT_LEN, INT_FEATURES_SIZE), dtype=np.int32)
    float_features = np.zeros((MAX_CONTEXT_LEN, FLOAT_FEATURES_SIZE), dtype=np.float32)
    for i in range(len(elem[1])):
        if elem[2][i][0]:
            int_features[i, 0] = 1
        if elem[2][i][1]:
            int_features[i, 1] = 1
        if elem[2][i][2]:
            int_features[i, 2] = 1
        float_features[i, 0] = elem[2][i][3]
        int_features[i, 3] = TAG2I[elem[3][i]]
        int_features[i, 4] = TAG2I[elem[4][i]]
    return int_features, float_features
# get_features

def make_comfort_data(data, tokenized_data):
    comfort_data = init_comfort_data(data)
    context, context_len, context_int_features, context_float_features = comfort_data[0]
    question, question_len = comfort_data[1]
    answer_begin, answer_end = comfort_data[2]

    for i in range(len(data)):
        context_len[i] = len(data[i][1])
        question_len[i] = len(data[i][5])
        context[i,:len(data[i][1])] = np.array(data[i][1], dtype=np.int32)
        question_context[i,:len(data[i][5])] = np.array(data[i][5], dtype=np.int32)
        context_int_features, context_float_features = get_features(data[i])
        answer_

    context_data = (context, context_len, context_float_features, context_float_features)
    question_data = (question, question_len)
    answer_data = (answer_begin, answer_end)
    comfort_data = (context_data, question_data, answer_data)
    return comfort_data
# make_comfort_data

def main():
    with open(FACEBOOK_DATA_PATH, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')

# main

if __name__ == "__main__":
    main()

