from constants import *

import numpy as np
import json
from tqdm import tqdm

def embed(tokens, w2i):
    emb = list()
    for t in tokens:
        emb.append(w2i[t])
    return emb

def embed_qa(qa, w2i):
    emb_qa = dict()
    emb_qa['question'] = embed(qa['question'], w2i)
    emb_qa['answer_begin'] = qa['answer_begin']
    emb_qa['answer_end'] = qa['answer_end']
    return emb_qa

def embed_paragraph(par, w2i):
    emb_par = dict()
    emb_par['context'] = embed(par['context'], w2i)
    emb_par['qas'] = list()
    for qa in par['qas']:
        emb_par['qas'].append(embed_qa(qa, w2i))
    return emb_par
# embed_paragraph

def embed_data(data, w2i):
    emb_train = dict()
    emb_train['paragraphs'] = list()
    for par in tqdm(data[0]['paragraphs']):
        emb_train['paragraph'].append(embed_paragraph(par, w2i))
    emb_test = dict()
    emb_test['paragraphs'] = list()
    for par in tqdm(data[1]['paragraphs']):
        emb_test['paragraph'].append(embed_paragraph(par, w2i))
    return emb_train, emb_test
# embed_data

def main():
    i2w = np.load(INDEX2WORD_PATH)
    w2i = dict()
    for i in range(len(i2w)):
        w2i[i2w[i]] = i
    data = json.load(open(TOKENIZED_TEST_PATH, 'r'))

    emb_data = dict()
    emb_data['paragraphs'] = list()

    for par in tqdm(data['paragraphs']):
        emb_data['paragraphs'].append(embed_paragraph(par, w2i))

    with open(EMBEDDED_TEST_PATH, 'w') as out:
        json.dump(emb_data, out, indent='  ')

if __name__ == "__main__":
    main()

