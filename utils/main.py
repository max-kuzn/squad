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
    test = 0
    m.train_model(sess, train, 0, batch_size=200, steps=10000)

if __name__ == "__main__":
    main()

