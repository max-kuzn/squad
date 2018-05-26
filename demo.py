import sys
sys.path.insert(0, 'utils')

from constants import *
from util import *
from model import *

import tensorflow as tf
import numpy as np
import msgpack
import spacy
import re
import unicodedata

###

with open(F_EMBEDDING_PATH, 'rb') as f:
    meta = msgpack.load(f, encoding='utf8')

I2TOKEN = meta['vocab']
TOKEN2I = dict()
for i in range(len(I2TOKEN)):
    TOKEN2I[I2TOKEN[i]] = i

tokenize = spacy.load('en', parser=False)

###

def get_tokens(text):
    global tokenize
    tokens = tokenize(text)
    tokens = [unicodedata.normalize('NFD', str(w)) for w in tokens]
    print(tokens)
    return tokens
# get_tokens

def get_indexes(tokens):
    indexes = np.zeros((1, len(tokens)), dtype=np.int32)
    for i in range(len(tokens)):
        if tokens[i] in TOKEN2I:
            indexes[0, i] = TOKEN2I[tokens[i]]
        else:
            indexes[0, i] = 1
    return indexes
# get_indexes

def prepare_data(context_txt, question_txt):
    context_tokens = get_tokens(context_txt)
    context = get_indexes(context_tokens)
    c_l = context.shape[1]
    context_len = np.zeros((1,), dtype=np.int32)
    context_len[0] = c_l
    #
    question_tokens = get_tokens(question_txt)
    question = get_indexes(question_tokens)
    q_l = question.shape[1]
    question_len = np.zeros((1,), dtype=np.int32)
    question_len[0] = q_l
    #TODO
    context_int_features = np.zeros((1, c_l, 5), dtype=np.int32)
    context_float_features = np.zeros((1, c_l, 1), dtype=np.float32)
    # Cork
    true_answer_begin = np.zeros((1,), dtype=np.int32)
    true_answer_end = np.zeros((1,), dtype=np.int32)
    return (
            (context, context_len),
            (context_int_features, context_float_features),
            (question, question_len),
            (true_answer_begin, true_answer_end)
            ), context_tokens
# prepare_data

def extract_answer(context_tokens, answer_begin, answer_end):
    answer = ''
    for i in range(answer_begin, answer_end):
        answer = answer + context_tokens[i] + ' '
    return answer
# extract_answer

def find_answer(sess, model, context, question):
    data, context_token = prepare_data(context, question)
    answer_begin, answer_end = model.get_answer(sess, data, window=15)
    answer_begin = answer_begin[0]
    answer_end = answer_end[0]
    return extract_answer(context_token, answer_begin, answer_end)
# find_answer

def main():
    print("Loading model...")
    m = Model()
    print("Done.")
    sess = tf.Session(graph=m.graph)
    m.load_model(sess)
    try:
        while(1):
            context = input("Conetxt:\n")
            question = input("Question:\n")
            print("Context:", context)
            print("Question:", question)
            answer = find_answer(sess, m, context, question)
            print("Answer:", answer)
    except EOFError:
        pass
# main

if __name__ == "__main__":
    main()

