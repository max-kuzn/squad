from constants import *
from util import *
from model import *

import tensorflow as tf

'''
def split_data(data, test_part=0.05):
    N = data[0][0].shape[0]
    test_N = int(N * test_part)

# split_data
'''

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
            epochs=1,
            batch_size=200,
            test_every=50
    )
    m.save_model(sess)
# main

if __name__ == "__main__":
    main()

