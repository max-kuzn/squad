from constants import *
from util import *
from model import *

import tensorflow as tf

def main():
    print("Loading data...")
    train = load_train()
    test = load_test()
    print("Done.\n")
    m = Model()
    sess = tf.Session(graph=m.graph)
    m.init_variables(sess)
    print("\n\n")
    print("Start train")
    m.train_model(
            sess,
            train,
            test,
            epochs=30,
            batch_size=32,
            train_summary_every=5,
            test_summary_every=10
    )
    m.save_model(sess)
# main

if __name__ == "__main__":
    main()

