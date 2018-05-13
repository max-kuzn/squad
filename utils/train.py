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
    print("Loading train data...")
    train = load_train()
    print("Done.\n")
    # train, test = split_data(data)
    m = Model()
    sess = tf.Session(graph=m.graph)
    m.init_variables(sess)
    print("\n\n")
    print("Start train")
    test = 0
    m.train_model(
            sess,
            train,
            0,
            steps=100,
            batch_size=200,
            test_every=1000
    )
    m.save_model(sess)
# main

if __name__ == "__main__":
    main()

