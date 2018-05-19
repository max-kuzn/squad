from constants import *
from util import *

from tqdm import tqdm
import numpy as np
import tensorflow as tf

BATCH_SIZE = 200
EPOCHS = 10
EMBEDDING_SIZE = 300
RNN_HIDDEN_SIZE = 256
SOFT = 2

def rnn_cell(hidden_size, name):
    return tf.nn.rnn_cell.LSTMCell(hidden_size, name=name)
# rnn_cell

def bidirect_cell(
        inputs,
        inputs_len,
        name,
        hidden_size=RNN_HIDDEN_SIZE,
        initial_state=None,
):
    return tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell(hidden_size, name=name+"_fw"),
            cell_bw=rnn_cell(hidden_size, name=name+"_bw"),
            inputs=inputs,
            sequence_length=inputs_len,
            initial_state_fw=initial_state,
            initial_state_bw=initial_state,
            dtype=tf.float32
        )
# bidirect_cell

def group_outputs(outputs, mode=None, bn=False):
    if mode == "concat":
        outputs = tf.concat([outputs[0], outputs[1]], axis=2)
    elif mode == "sum":
        outputs = tf.add(outputs[0], outputs[1])
    else:
        raise
    if bn:
        outputs = tf.layers.batch_normalization(outputs)
    return outputs
# group_outputs

def group_LSTM_state(states, make=False, mode=None, bn=False):
    if make and (mode == "concat_all" or mode == "sum_all"):
        raise
    if mode == "concat":
        states = tf.concat(states, axis=2)
    elif mode == "concat_all":
        states = tf.concat([states[0], states[1]], axis=2)
        states = tf.concat([states[0], states[1]], axis=1)
    elif mode == "sum":
        states = tf.add(states[0], states[1])
    elif mode == "sum_all":
        states = tf.add(states[0], states[1])
        states = tf.add(states[0], states[1])
    else:
        raise
    if bn:
        states = tf.layers.batch_normalization(states)
    if make:
        return tf.contrib.rnn.LSTMStateTuple(states[0], states[1])
    return states
# make_LSTM_state

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
                question_out = self.__setup_question()
            with tf.variable_scope("context"):
                context_out = self.__setup_context(
                        question_out
                )
            self.__setup_loss_and_answers(
                    batch_size,
                    question_out,
                    context_out
            )
            self.__summary_all()

            optimizer = tf.train.RMSPropOptimizer(0.0001)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 1000)
            self.train_step = optimizer.apply_gradients(zip(gradients, variables))
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
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
        self.true_answer_begin = tf.placeholder(
            tf.int32,
            name="true_answer_begin",
            shape=(None)
        )
        self.true_answer_end = tf.placeholder(
            tf.int32,
            name="true_answer_end",
            shape=(None)
        )
    # __setup_inputs

    def __setup_question(self):
        question_layer1 = bidirect_cell(
                self.question,
                self.question_len,
                name="q_rnn1",
                hidden_size=RNN_HIDDEN_SIZE
        )
        output1 = group_outputs(
                question_layer1[0],
                mode="concat",
                bn=False
        )
        '''
        stat1 = group_LSTM_state(
                question_layer1[1],
                make=True,
                mode="sum",
                bn=False
        )
        '''
        question_layer2 = bidirect_cell(
                output1,
                self.question_len,
                name="q_rnn2",
                hidden_size=RNN_HIDDEN_SIZE,
        #        initial_state=stat1
        )
        return question_layer2
    # __setup_question

    def __setup_context(self, question_out):
        question_state = group_LSTM_state(
                question_out[1],
                make=False,
                mode="concat",
                bn=True
        )
        question_state = tf.layers.dense(
                question_state,
                units=RNN_HIDDEN_SIZE
        )
        start_state = tf.contrib.rnn.LSTMStateTuple(
                question_state[0],
                question_state[1]
        )
        context_layer1 = bidirect_cell(
                self.context,
                self.context_len,
                name="c_rnn1",
                hidden_size=RNN_HIDDEN_SIZE
        )
        context_layer2 = bidirect_cell(
                group_outputs(context_layer1[0], mode="concat", bn=True),
                self.context_len,
                name="c_rnn2",
                hidden_size=RNN_HIDDEN_SIZE,
                initial_state=start_state
        )
        context_layer3 = bidirect_cell(
                group_outputs(context_layer2[0], mode="concat", bn=True),
                self.context_len,
                name="c_rnn3",
                hidden_size=RNN_HIDDEN_SIZE,
                initial_state=start_state
        )
        return context_layer3
    # __setup_context()

    def __setup_loss_and_answers(self, batch_size,
            question_out,
            context_out
    ):
        context_outputs = group_outputs(context_out[0], mode="concat", bn=True)
        dense_begin = tf.layers.dense(
                inputs=context_outputs,
                units=RNN_HIDDEN_SIZE,
                use_bias=True,
                name='dense_begin'
        )
        dense_end = tf.layers.dense(
                inputs=context_outputs,
                units=RNN_HIDDEN_SIZE,
                use_bias=True,
                name='dense_end'
        )
        question_state = group_LSTM_state(
                question_out[1],
                mode="concat_all"
        )
        question_state = tf.reshape(
                question_state,
                (tf.shape(question_state)[0], tf.shape(question_state)[1], 1)
        )
        points_begin = tf.matmul(
                dense_begin,
                question_state,
                name='points_begin'
        )
        points_end = tf.matmul(
                dense_end,
                question_state,
                name='points_end'
        )
        points_begin = tf.reshape(points_begin, tf.shape(points_begin)[:-1])
        points_end = tf.reshape(points_end, tf.shape(points_end)[:-1])
        softmax_begin = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.true_answer_begin,
                logits=points_begin,
                name="softmax_begin"
        )
        softmax_end = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.true_answer_end,
                logits=points_end,
                name="softmax_end"
        )
        loss_begin = tf.reduce_mean(softmax_begin, name='loss_begin')
        loss_end = tf.reduce_mean(softmax_end, name='loss_end')
        self.loss = tf.add(loss_begin, loss_end, name='loss')
        self.answer_begin = tf.argmax(
                points_begin,
                name='answer_begin',
                axis=-1,
                output_type=tf.int32
        )
        self.answer_end = tf.argmax(
                points_end,
                name='answer_end',
                axis=-1,
                output_type=tf.int32
        )
    # __setup_loss_and_answers

    def __summary_all(self):
        tf.summary.scalar('loss', self.loss)
        good_begin = tf.equal(self.true_answer_begin, self.answer_begin)
        good_end = tf.equal(self.true_answer_end, self.answer_end)
        tf.summary.scalar('answer begin accuracy',
            tf.count_nonzero(
                good_begin,
                dtype=tf.float32
            ) / tf.cast(tf.size(good_begin), tf.float32)
        )
        '''
        tf.summary.scalar('answer end accuracy',
            tf.count_nonzero(
                good_end,
                dtype=tf.float32
            ) * 100 / tf.cast(tf.size(good_end), dtype=tf.float32)
        )
        tf.summary.scalar('answer begin or end accuracy',
            tf.count_nonzero(
                tf.logical_or(good_begin, good_end),
                dtype=tf.float32
            ) * 100 / tf.cast(tf.size(good_end), dtype=tf.float32)
        )
        '''
        tf.summary.scalar('answer begin and end accuracy',
            tf.count_nonzero(
                tf.logical_and(good_begin, good_end),
                dtype=tf.float32
            ) / tf.cast(tf.size(good_begin), tf.float32)
        )
        # f1 score
        TP = tf.nn.relu(
            tf.subtract(
                tf.minimum(self.answer_end, self.true_answer_end),
                tf.maximum(self.answer_begin, self.true_answer_begin)
            ),
            name='TP'
        )
        FPN = tf.nn.relu(
            tf.subtract(
                tf.subtract(
                    tf.maximum(self.answer_end, self.true_answer_end),
                    tf.minimum(self.answer_begin, self.true_answer_begin)
                ),
                TP
            ),
            name='FPN'
        )
        tf.summary.scalar('F1 score',
            tf.reduce_mean(
                2 * TP / (2 * TP + FPN)
            )
        )
        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(
                LOGS_PATH + SEP + "train",
                graph=self.graph
        )
        self.test_writer = tf.summary.FileWriter(
                LOGS_PATH + SEP + "test",
                graph=self.graph
        )
    # __summary_all

    def init_variables(self, session):
        if self.__logs:
            print("Start initialize variables...")
        session.run(self.init)
        if self.__logs:
            print("Done.")
    # init_variables

    def train_model(
            self,
            session,
            train,
            test,
            batch_size=BATCH_SIZE,
            train_summary_every=None,
            test_summary_every=None,
            epochs=1
    ):
        step = 0
        for e in range(epochs):
            for batch in tqdm(next_batch(train, batch_size, self.__embeddings)):
                if train_summary_every != None and step % train_summary_every == 0:
                    summary, _ = session.run(
                            [
                                self.summary,
                                self.train_step
                            ],
                            {
                                self.context: batch[0][0],
                                self.context_len: batch[0][1],
                                self.question: batch[1][0],
                                self.question_len: batch[1][1],
                                self.true_answer_begin: batch[2][0],
                                self.true_answer_end: batch[2][1]
                            }
                         )
                    self.train_writer.add_summary(summary, step)
                else:
                    session.run(
                            self.train_step,
                            {
                                self.context: batch[0][0],
                                self.context_len: batch[0][1],
                                self.question: batch[1][0],
                                self.question_len: batch[1][1],
                                self.true_answer_begin: batch[2][0],
                                self.true_answer_end: batch[2][1]
                            }
                    )
                if test_summary_every != None and step % test_summary_every == 0:
                    self.evaluate(session, test, step, batch_size=BATCH_SIZE)
                step += 1
    # train_model

    def evaluate(self, session, test, x, batch_size=BATCH_SIZE):
        batch = get_random_batch(test, batch_size, self.__embeddings)
        summary = session.run(
            self.summary,
            {
                self.context: batch[0][0],
                self.context_len: batch[0][1],
                self.question: batch[1][0],
                self.question_len: batch[1][1],
                self.true_answer_begin: batch[2][0],
                self.true_answer_end: batch[2][1]
            }
        )
        self.test_writer.add_summary(summary, x)
    # evaluate

    def save_model(self, session, path=MODEL_PATH):
        self.saver.save(session, path)
    # save_movel

    def load_model(self, session, path=MODEL_PATH):
        self.saver.restore(session, path)
    # load_model

# Model

