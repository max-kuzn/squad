from constants import *
from util import *
from model import *

import tensorflow as tf

def main():
    print("Loading train data...")
    train = load_train()
    print("Done.\n")
    m = Model()
    sess = tf.Session(graph=m.graph)
    m.init_variables(sess)
    print("\n\n")
    print("Start train")
    m.train_model(sess, train, 200, 10)


if __name__ == "__main__":
    main()

