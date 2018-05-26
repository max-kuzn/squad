import sys
sys.path.insert(0, 'utils')

from constants import *
from util import *
from model import *

import tensorflow as tf

def main():
    print("Loading test data...")
    test = load_test()
    print("Done.\n")
    m = Model()
    sess = tf.Session(graph=m.graph)
    print("\n\n")
    print("Loadding model.")
    print("Start testing")
    m.load_model(sess)
    m.evaluate(sess,
            test,
            window=15,
            keep_prob=1.0,
            batch_size=32,
            tensorboard=False
    )

# main

if __name__ == "__main__":
    main()

