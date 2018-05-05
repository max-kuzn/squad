from constants import DATA_PATH
from tqdm import tqdm
import numpy as np

def main():
    f = open(DATA_PATH + "wiki-news-300d-1M-subword.vec", 'r')
    voc_size, emb_size = map(int, f.readline().split())
    vocabulary = np.chararray((voc_size, ), unicode=True)
    emb = np.empty((voc_size, emb_size), dtype=np.float32)

    for i in tqdm(range(voc_size)):
        line_split = f.readline().split()
        vocabulary[i] = line_split[0]
        j = 0
        for x in line_split[1:]:
            emb[i, j] = float(x)
            j += 1

    np.save(DATA_PATH + '/' + "voc.npy", vocabulary)
    np.save(DATA_PATH + '/' + "emb.npy", emb)
#main

if __name__ == "__main__":
    main()

