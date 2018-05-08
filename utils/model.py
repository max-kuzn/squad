from constants import *
from utils import *

from tqdm import tqdm
import numpy as np
import tensorflow as tf

BATCH_SIZE = 32
EPOCHS = 10
EMBEDDING_SIZE = 300
RNN_HIDDEN_SIZE = 128

def rnn_cell():
    return tf.nn.rnn_cell.LSTMCell(RNN_HIDDEN_SIZE)
# rnn_cell

def bidirect_cell(inputs, inputs_len, initial_state=None):
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell(),
            cell_bw=rnn_cell(),
            inputs=inputs,
            sequence_length=inputs_len,
            initial_state_fw=initial_state,
            initial_state_bw=initial_state,
            dtype=tf.float32
        )
    output_fw, output_bw = outputs
    state_fw, state_bw = states
    output = tf.concat([output_fw, output_bw], axis=-1)
    state = tf.concat([state_fw, state_bw], axis=-1)
    return output, state
# bidirect_cell

class Model:
    def __init__(
            self,
            mode="train" # "train", "test" or "both"
            logs=True
            batch_size=BATCH_SIZE
            epochs=EPOCHS,
            embedding_size=EMBEDDING_SIZE
    ):
        self.__setup_constants(
                bs=batch_size,
                e=epochs,
                ez=embedding_size
        )
        self.__logs = logs
        self.__mode = mode
        self.__load_embeddings()
        self.__setup_model()
    # __init__

    def __setup_constants(self, bs, e, es):
        self.__BATCH_SIZE = bs
        self.__EPOCHS = e
        self.__EMBEDDING_SIZE = es
    # __setup_constants

    def __load_embeddings(self):
        if self.__logs:
            print("Start loading embeddings in mode =" + self.__mode \
                    + "...")
        self.__embeddings = Embedding(mode=self.__mode)
        if self.__logs:
            print("Done.")
    # __load_embeddings

    def __setup_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope("input"):
                self.__setup_inputs()
            embedding = tf.
            with tf.variable_scope("question"):
                q_outs = self.__setup_question()
                question_state = q_outs[#TODO]
                question_
            with tf.va
        # TODO
    # __setup_model

    def __setup_inputs():
        self.context = tf.placeholder(
            tf.int32,
            name="context",
            shape=(None, None)
        )
        self.question = tf.placeholder(
            tf.int32,
            name="question",
            shape=(None, None)
        )
        self.context_len = tf.placeholder(
            tf.int32,
            name="context_lenght",
            shape=(None)
        )
        self.question_len = tf.placeholder(
            tf.int32,
            name="question_lenght",
            shape=(None)
        )
        self.answer_begin = ft.placeholder(
            tf.int32,
            name="answer_begin",
            shape=(None)
        )
        self.answer_end = tf.placeholder(
            tf.int32,
            name="answer_end",
            shape=(None)
        )
    # __setup_inputs

    def init_variables(self, session):
        if self.__logs:
            print("Start initialize variables...")
        session.run(self.__init)
        if self.__logs:
            print("Done.")
    # init_variables

    def train_model(self, session, train, epochs=EPOCHS):
    # train_model

    def save_model(self, session, path=MODEL_PATH):
    # save_movel

    def load_model(self, session, path=MODEL_PATH):
    # load_model

# Model
