import sys
sys.path.insert(0, 'utils')

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
    print("Initing variables...")
    m.init_variables(sess)
    print("Done.\n")
    print("\n\n")
    print("Start train")
    m.train_model(
            sess,
            train,
            test,
            epochs=7,
            steps=None,
            batch_size=32,
            keep_prob=0.7,
            window=15,
            train_summary_every=50,
            test_summary_every=200
    )
    print("Start testing...")
    m.evaluate(sess,
            test,
            window=15,
            keep_prob=1.0,
            batch_size=32,
            tensorboard=False
    )
    m.save_model(sess)
# main

if __name__ == "__main__":
    main()

