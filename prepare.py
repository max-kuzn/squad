import sys
sys.path.insert(0, '../utils')

from constants import *

import numpy as np
from tqdm import tqdm
import msgpack

INT_FEATURES_SIZE = 5
FLOAT_FEATURES_SIZE = 1
MAX_CONTEXT_LEN = 0

def save_comfort_data(comfort_data):
    np.savez(F_COMFORT_TRAIN_PATH,
            context          = comfort_data[0][0][0],
            context_len      = comfort_data[0][0][1],
            context_int_features = comfort_data[0][1][0],
            context_float_features = comfort_data[0][1][1],
            question         = comfort_data[0][2][0],
            question_len     = comfort_data[0][2][1],
            answer_begin     = comfort_data[0][3][0],
            answer_end       = comfort_data[0][3][1]
    )
    np.savez(F_COMFORT_TEST_PATH,
            context          = comfort_data[1][0][0],
            context_len      = comfort_data[1][0][1],
            context_int_features = comfort_data[1][1][0],
            context_float_features = comfort_data[1][1][1],
            question         = comfort_data[1][2][0],
            question_len     = comfort_data[1][2][1],
            answer_begin     = comfort_data[1][3][0],
            answer_end       = comfort_data[1][3][1]
    )
# save_comfort_data

def init_comfort_data(data):
    global MAX_CONTEXT_LEN
    max_context_len = 0
    max_question_len = 0
    n = len(data)
    for elem in data:
        max_context_len = max(max_context_len, len(elem[1]))
        max_question_len = max(max_question_len, len(elem[5]))
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
        int_features[i, 3] = elem[3][i] # tag
        int_features[i, 4] = elem[4][i] # ent
    return int_features, float_features
# get_features

def find_dev_answer(par, answer, tokens):
    ind = par.find(answer)
    begin = len(tokens)
    end = len(tokens)
    find_b = False
    find_e = False
    for i in range(len(tokens)):
        if tokens[i][0] == ind:
            find_b = True
            begin = i
            break
    for i in range(begin, len(tokens)):
        if abs(tokens[i][1] - (ind + len(answer))) <= 0:
            find_e = True
            end = i
            break
    return begin, end, find_b and find_e
# find_dev_answer

def make_comfort_data(data, is_dev=False):
    comfort_data = init_comfort_data(data)
    context, context_len, context_int_features, context_float_features = comfort_data[0]
    question, question_len = comfort_data[1]
    answer_begin, answer_end = comfort_data[2]

    bad = list()
    for i in tqdm(range(len(data))):
        context_len[i] = len(data[i][1])
        question_len[i] = len(data[i][5])
        context[i,:len(data[i][1])] = np.array(data[i][1], dtype=np.int32)
        question[i,:len(data[i][5])] = np.array(data[i][5], dtype=np.int32)
        context_int_features[i], context_float_features[i] = get_features(data[i])
        if is_dev:
            answer_begin[i], answer_end[i], good = find_dev_answer(
                    data[i][6], data[i][8][0], data[i][7]
            )
            if not good:
                bad.append(i)
        else:
            answer_begin[i] = data[i][8]
            answer_end[i] = data[i][9]

    good = np.ones((context.shape[0],), dtype=np.bool)
    good[np.array(bad, np.int)] = False

    context_data = (context[good], context_len[good])
    context_features = (context_int_features[good], context_float_features[good])
    question_data = (question[good], question_len[good])
    answer_data = (answer_begin[good], answer_end[good])
    comfort_data = (context_data, context_features, question_data, answer_data)
    return comfort_data
# make_comfort_data

def main():
    print("Loading data...")
    with open(F_DATA_PATH, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    print("Done.")
    print("Processing...")
    data = (
            make_comfort_data(data['train']),
            make_comfort_data(data['dev'], is_dev=True)
    )
    print("Done.")
    print("Saving...")
    save_comfort_data(data)
    print("Done.")
# main

if __name__ == "__main__":
    main()

