from constants import *

import numpy as np
import json
from tqdm import tqdm
import nltk

###

I2PARTS = ['VBZ', "''", 'LS', ',', 'JJR', 'JJS', 'RB', 'JJ', 'VB',
        'NN', 'WRB', 'POS', 'RP', 'PRP$', 'NNPS', 'RBR', 'SYM', 'VBN',
        'UH', 'TO', 'VBP', '.', ')', ':', 'CD', '(', 'IN', '$', 'CC',
        'VBD', 'RBS', 'DT', 'WP', 'PRP', 'WP$', 'NNS', 'VBG', 'FW',
        'PDT', 'EX', 'WDT', '#', 'MD', 'NNP'
        ]
PARTS_NUM = len(I2PARTS)
PARTS2I = dict()
for i in range(PARTS_NUM):
    PARTS2I[I2PARTS[i]] = i

FEATURES_SIZE = 1 + 1
MAX_CONTEXT_LEN = 0

###

def load_tokenized_data():
    train = json.load(open(TOKENIZED_TRAIN_PATH, 'r'))
    test = json.load(open(TOKENIZED_TEST_PATH, 'r'))
    return train, test
# load_tokenized_data

def save_comfort_data(comfort_data):
    np.savez(COMFORT_TRAIN_PATH,
            context          = comfort_data[0][0][0],
            context_len      = comfort_data[0][0][1],
            context_features = comfort_data[0][0][2],
            question         = comfort_data[0][1][0],
            question_len     = comfort_data[0][1][1],
            answer_begin     = comfort_data[0][2][0],
            answer_end       = comfort_data[0][2][1]
    )
    np.savez(COMFORT_TEST_PATH,
            context          = comfort_data[1][0][0],
            context_len      = comfort_data[1][0][1],
            context_features = comfort_data[1][0][2],
            question         = comfort_data[1][1][0],
            question_len     = comfort_data[1][1][1],
            answer_begin     = comfort_data[1][2][0],
            answer_end       = comfort_data[1][2][1]
    )
# save_comfort_data

def question_embed(tokens, w2i):
    emb = list()
    for t in tokens:
        emb.append(w2i[t])
    return emb
# question_embed

def context_embed(tokens, w2i):
    return question_embed(tokens, w2i)
# context_embed

def embed_qa(qa, w2i):
    emb_qa = dict()
    emb_qa['question'] = question_embed(qa['question'], w2i)
    emb_qa['answer_begin'] = qa['answer_begin']
    emb_qa['answer_end'] = qa['answer_end']
    return emb_qa
# embed_qa

def embed_paragraph(par, w2i):
    emb_par = dict()
    emb_par['context'] = context_embed(par['context'], w2i)
    emb_par['qas'] = list()
    for qa in par['qas']:
        emb_par['qas'].append(embed_qa(qa, w2i))
    return emb_par
# embed_paragraph

def embed_data(data, w2i):
    emb_data = dict()
    emb_data['paragraphs'] = list()
    for par in tqdm(data['paragraphs']):
        emb_data['paragraphs'].append(embed_paragraph(par, w2i))
    return emb_data
# embed_data

def init_comfort_data(data):
    global MAX_CONTEXT_LEN
    n = 0
    max_context_len = 0
    max_question_len = 0
    for p in data['paragraphs']:
        if len(p['context']) > max_context_len:
            max_context_len = len(p['context'])
        for q in p['qas']:
            n += 1
            if len(q['question']) > max_question_len:
                max_question_len = len(q['question'])
    MAX_CONTEXT_LEN = max_context_len
    context = np.zeros((n, max_context_len), dtype=np.int32)
    context_len = np.zeros((n,), dtype=np.int32)
    context_features = np.zeros((n, max_context_len, FEATURES_SIZE), dtype=np.int32)
    question = np.zeros((n, max_question_len), dtype=np.int32)
    question_len = np.zeros((n,), dtype=np.int32)
    answer_begin = np.zeros((n,), dtype=np.int32)
    answer_end = np.zeros((n,), dtype=np.int32)
    return (
            (context, context_len, context_features),
            (question, question_len),
            (answer_begin, answer_end)
           )
# init_comfort_data

def get_features(context, question):
    features = np.zeros((MAX_CONTEXT_LEN, FEATURES_SIZE), dtype=np.int32)
    for i in range(len(context)):
        part = nltk.pos_tag([context[i]])[0][1]
        features[i, 0] = PARTS2I[part]
        if context[i] in question:
            features[i, 1] = 1
    return features
# get_features

def make_comfort_data(data, tokenized_data):
    comfort_data = init_comfort_data(data)
    context, context_len, context_features = comfort_data[0]
    question, question_len = comfort_data[1]
    answer_begin, answer_end = comfort_data[2]
    
    i = 0
    par_num = 0
    for p in tqdm(data['paragraphs']):
        qa_num = 0
        for q in p['qas']:
            context_len[i] = len(p['context'])
            question_len[i] = len(q['question'])
            context[i, :context_len[i]] = \
                np.array(p['context'], dtype=np.int32)
            question[i, :question_len[i]] = \
                np.array(q['question'], dtype=np.int32)
            answer_begin[i] = q['answer_begin']
            answer_end[i] = q['answer_end']
            context_features[i] = get_features(
                    tokenized_data['paragraphs'][par_num]['context'],
                    tokenized_data['paragraphs'][par_num]['qas'][qa_num]
            )
            i += 1
            qa_num += 1
        par_num += 1
    context_data = (context, context_len, context_features)
    question_data = (question, question_len)
    answer_data = (answer_begin, answer_end)
    comfort_data = (context_data, question_data, answer_data)
    return comfort_data
# make_comfort_data

def main():
    i2w = np.load(INDEX2WORD_PATH)
    w2i = dict()
    for i in range(len(i2w)):
        w2i[i2w[i]] = i
    print("Loading data...")
    tokenized_data = load_tokenized_data()
    print("Done.")
    print("Embedding data...")
    data = (embed_data(tokenized_data[0], w2i),
            embed_data(tokenized_data[1], w2i)
        )
    print("Done.")
    print("Making comfort data...")
    comfort_data = (make_comfort_data(data[0], tokenized_data[0]),
            make_comfort_data(data[1], tokenized_data[1])
        )
    print("Done.")
    print("Saving data...")
    save_comfort_data(comfort_data)
    print("Done.")
# main

if __name__ == "__main__":
    main()

