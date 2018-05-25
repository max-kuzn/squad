import sys
sys.path.insert(0, '../utils')

from constants import *
from util import tokenize

import subprocess
import json
from tqdm import tqdm

all_qas = 0
good_qas = 0

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


def find_subtext(text, sub):
    global all_qas
    global good_qas
    all_qas += 1
    for i in range(len(text) - len(sub) + 1):
        good = True
        for j in range(len(sub)):
            if text[i + j] != sub[j]:
                good = False
                break
        if good:
            good_qas += 1
            return (i, i + len(sub))
    raise NameError("Can't find answer")
#find_subtext

def tokenize_qa(qa, context):
    tokenized_qa = list()
    a = qa["answers"][0]
    question_answer = dict()
    question_answer["question"] = tokenize(qa["question"])
    question_answer["answer_begin"], question_answer["answer_end"] = \
        find_subtext(context, tokenize(a["text"]))
    return question_answer
#tokenize_qa

def tokenize_paragraph(par):
    tok_par = dict()
    tok_par['context'] = tokenize(par['context'])
    tok_par['qas'] = list()
    for qas in par['qas']:
        try:
            tok_par['qas'].append(tokenize_qa(qas, tok_par["context"]))
        except NameError as ne:
            #print(ne)
            ''
    return tok_par
#tokenize_paragraph

def main():
    jdata = json.load(open(RAW_TEST_PATH, 'r'))

    par = jdata['data'][0]['paragraphs'][0]
    tokenize_data = dict()
    tokenize_data['paragraphs'] = list()

    for data in tqdm(jdata['data']):
        for par in data['paragraphs']:
            tokenize_data['paragraphs'].append(tokenize_paragraph(par))
    with open(TOKENIZED_TEST_PATH, "w") as out:
        json.dump(tokenize_data, out, indent='  ')
    print(str(good_qas) + '/' + str(all_qas))
    print(good_qas / all_qas * 100)
#main

if __name__ == "__main__":
    main()

