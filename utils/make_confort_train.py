from constants import *

import json
import numpy as np
from tqdm import tqdm

def main():
    data = json.load(open(EMBEDDED_TRAIN_PATH))
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
    np.savez(COMFORT_TRAIN_PATH,
            context=context, context_len=context_len,
            question=question, question_len=question_len,
            answer_begin=answer_begin, answer_end=answer_end
    )
# main

if __name__ == "__main__":
    main()

