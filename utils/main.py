from constants import *
from util import *
from model import *

import tensorflow as tf

def main():
    m = Model()
    sess = tf.Session()
    m.init_variables(sess)

if __name__ == "__main__":
    main()

