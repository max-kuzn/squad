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
    print("\n\n")
    print("Loadding model.")
    sess = tf.Session(graph=m.graph)
    m.load_model(sess, '/home/max/programming/squad/squad/data/model/saved')
    print("Start testing", test[0][0].shape[0] // 32 + int(test[0][0].shape[0] != 32), "batches")

    f1 = m.evaluate(sess,
            test,
            window=15,
            keep_prob=1.0,
            batch_size=32,
            tensorboard=False
    )
    print(f1)
# main

if __name__ == "__main__":
    main()

