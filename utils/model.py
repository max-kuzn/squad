from constants import *
from util import *

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
    out = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell(),
            cell_bw=rnn_cell(),
            inputs=inputs,
            sequence_length=inputs_len,
            initial_state_fw=initial_state,
            initial_state_bw=initial_state,
            dtype=tf.float32
        )
    outputs, states = out
    #TODO
    return tf.concat(outputs, -1), tf.concat(states, -1)
# bidirect_cell

class Model:
    def __init__(
            self,
            mode="train", # "train", "test" or "both"
            logs=True,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            embedding_size=EMBEDDING_SIZE
    ):
        self.__setup_constants(
                bs=batch_size,
                e=epochs,
                es=embedding_size
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
            print("Start loading embeddings in mode = \"" + self.__mode \
                    + "\"...")
        self.__embeddings = Embedding(mode=self.__mode)
        if self.__logs:
            print("Done.")
    # __load_embeddings

    def __setup_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope("input"):
                self.__setup_inputs()
            batch_size = tf.shape(self.question)[0]
            question_size = tf.shape(self.question)[1]
            context_size = tf.shape(self.context)[1]
            with tf.variable_scope("question"):
                question_outputs, question_state = self.__setup_question()
            with tf.variable_scope("context"):
                context_outputs, context_state = self.__setup_context(
                        question_outputs, question_state
                )
            self.loss = self.__get_loss(batch_size,
                    question_outputs, question_state,
                    context_outputs, context_state
            )
            tf.summary.scalar('loss', self.loss)
            optimizer = tf.train.AdamOptimizer(0.0001)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5000)
            self.train_step = optimizer.apply_gradients(zip(gradients, variables))
            self.init = tf.global_variables_initializer()
        self.writer = tf.summary.FileWriter(LOGS_PATH, graph=self.graph)
    # __setup_model

    def __setup_inputs(self):
        self.context = tf.placeholder(
            tf.float32,
            name="context",
            shape=(None, None, self.__EMBEDDING_SIZE)
        )
        self.question = tf.placeholder(
            tf.float32,
            name="question",
            shape=(None, None, self.__EMBEDDING_SIZE)
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
        self.answer_begin = tf.placeholder(
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

    def __setup_question(self):
        return bidirect_cell(self.question, self.question_len)
    # __setup_question

    def __setup_context(self, question_outputs, question_state):
        return bidirect_cell(self.context, self.context_len,
            initial_state=None #question_state
        )
    # __setup_context()

    def __get_loss(self, batch_size,
            question_outputs, question_state,
            context_outputs, context_state
    ):
        prob = tf.layers.dense(
                inputs=context_outputs,
                units=2*RNN_HIDDEN_SIZE,
                use_bias=True
        )
        # TODO
        prob = tf.matmul(prob,
                tf.reshape(question_state[0],
                    (batch_size, 2*RNN_HIDDEN_SIZE, 1)
                )
        )
        prob = tf.reshape(prob, tf.shape(prob)[:-1])
        print("_________________")
        print(prob)
        print(self.answer_begin)
        print("_________________")
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.answer_begin,
                logits=prob,
                name="softmax"
            )
        )
    # __get_loss

    def init_variables(self, session):
        if self.__logs:
            print("Start initialize variables...")
        session.run(self.init)
        if self.__logs:
            print("Done.")
    # init_variables

    def train_model(self, session, train, batch_size, epochs=EPOCHS):
        avg_loss = 0
        #TODO
        for i in range(EPOCHS * 1000):
            batch = get_batch(train, batch_size, self.__embeddings)
            context, context_len = batch[0]
            question, question_len = batch[1]
            answer_begin, answer_end = batch[2]
            loss, _ = session.run([self.loss, self.train_step],
                    {
                        self.context: context,
                        self.context_len: context_len,
                        self.question: question,
                        self.question_len: question_len,
                        self.answer_begin: answer_begin,
                        self.answer_end: answer_end
                    }
            )
            #TODO
            print(loss)
    # train_model

    def save_model(self, session, path=MODEL_PATH):
        #TODO
        return
    # save_movel

    def load_model(self, session, path=MODEL_PATH):
        #TODO
        return
    # load_model

# Model

