from constants import *

from tqdm import tqdm
import numpy as np
import fastText

def main():
    print("Loadding embeddings...")
    model = fastText.load_model(ALL_EMBEDDING_PATH)
    print("Done")
    i2w = np.load(INDEX2WORD_PATH)
    voc_size = i2w.shape[0]
    emb_size = model.get_dimension()
    train_emb = np.empty((voc_size, emb_size), dtype=np.float32)
    for i in tqdm(range(voc_size)):
        train_emb[i] = model.get_word_vector(i2w[i])
    np.save(TRAIN_EMBEDDING_PATH, train_emb)
    print("Shape =", train_emb.shape)
#main

if __name__ == "__main__":
    main()
