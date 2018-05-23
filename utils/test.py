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
    m.validate(sess, test)

# main

if __name__ == "__main__":
    main()

