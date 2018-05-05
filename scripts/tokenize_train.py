import json
from nltk.tokenize import RegexpTokenizer
from constants import RAW_TRAIN_PATH, TOKENIZED_TRAIN_PATH

all_qas = 0
good_qas = 0

'''
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer('\w+|\$[\d]+|\S+')
words = [w.lower() for w in tokenized if w.isalnum()]
'''

'''
def tokenize(text):
    return text
#tokenize
'''

def tokenize(vector):
    tokenizer = RegexpTokenizer('\w+')
    tokens = tokenizer.tokenize(vector)
    words = [w.lower() for w in tokens if w[0].isalnum()]
    return words


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
    for a in qa["answers"]:
        question_answer = dict()
        question_answer["question"] = tokenize(qa["question"])
        question_answer["answer_begin"], question_answer["answer_end"] = \
            find_subtext(context, tokenize(a["text"]))
        tokenized_qa.append(question_answer)
    return tokenized_qa
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
    jdata = json.load(open(RAW_TRAIN_PATH, 'r'))

    par = jdata['data'][0]['paragraphs'][0]
    tokenize_data = dict()
    tokenize_data['paragraphs'] = list()

    for data in jdata['data']:
        for par in data['paragraphs']:
            tokenize_data['paragraphs'].append(tokenize_paragraph(par))
    with open(TOKENIZED_TRAIN_PATH, "w") as out:
        json.dump(tokenize_data, out, indent='  ')
    print(str(good_qas) + '/' + str(all_qas))
    print(good_qas / all_qas * 100)
#main

if __name__ == "__main__":
    main()

