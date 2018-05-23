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
PARTS2I = dict()
for i in range(len(I2PARTS)):
    PARTS2I[I2PARTS[i]] = i

###

def load_tokenized_data():
    train = json.load(open(TOKENIZED_TRAIN_PATH, 'r'))
    test = json.load(open(TOKENIZED_TEST_PATH, 'r'))
    return train, test
# load_tokenized_data

def save_comfort_data(comfort_data)
    np.savez(COMFORT_TRAIN_PATH,
            context     =   comfort_data[0][0][0],
            context_len =   comfort_data[0][0][1],
            question    =   comfort_data[0][1][0],
            question_len=   comfort_data[0][1][1],
            answer_begin=   comfort_data[0][2][0],
            answer_end  =   comfort_data[0][2][1]
    )
    np.savez(COMFORT_TEST_PATH,
            context     =   comfort_data[1][0][0],
            context_len =   comfort_data[1][0][1],
            question    =   comfort_data[1][1][0],
            question_len=   comfort_data[1][1][1],
            answer_begin=   comfort_data[1][2][0],
            answer_end  =   comfort_data[1][2][1]
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
        emb_data['paragraph'].append(embed_paragraph(par, w2i))
    return emb_data
# embed_data

def blabla(data):
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
    print(n, max_context_len, max_question_len)
    context = np.zeros((n, max_context_len), dtype=np.int32)
    question = np.zeros((n, max_question_len), dtype=np.int32)
    context_len = np.empty((n,), dtype=np.int32)
    question_len = np.empty((n,), dtype=np.int32)
    answer_begin = np.empty((n,), dtype=np.int32)
    answer_end = np.empty((n,), dtype=np.int32)

    i = 0
    for p in tqdm(data['paragraphs']):
        for q in p['qas']:
            context_len[i] = len(p['context'])
            question_len[i] = len(q['question'])
            context[i, :context_len[i]] = \
                np.array(p['context'], dtype=np.int32)
            question[i, :question_len[i]] = \
                np.array(q['question'], dtype=np.int32)
            answer_begin[i] = q['answer_begin']
            answer_end[i] = q['answer_end']
            i += 1
    context_data = (context, context_len)
    question_data = (question, question_len)
    answer_data = (answer_begin, answer_end)
    comfort_data = (context_data, question_data, answer_data)
    return comfort_data
# blabla

def main():
    i2w = np.load(INDEX2WORD_PATH)
    w2i = dict()
    for i in range(len(i2w)):
        w2i[i2w[i]] = i
    data = load_tokenized_data()
    data = (embed_data(data[0], w2i), embed_data(data[1], w2i))
# main

if __name__ == "__main__":
    main()

