from constants import *
from util import *

from tqdm import tqdm
import numpy as np
import tensorflow as tf

BATCH_SIZE = 32
EMBEDDING_SIZE = 300
RNN_HIDDEN_SIZE = 128
INT_FEATURES_SIZE = 5
FLOAT_FEATURES_SIZE = 1
PARTS_NUM = 44
ENT_NUM = 19
TAG_NUM = 50

def rnn_cell(hidden_size, name, keep_in, keep_out, keep_state):
    return tf.nn.rnn_cell.DropoutWrapper(
            cell=tf.nn.rnn_cell.LSTMCell(hidden_size, name=name),
            input_keep_prob=keep_in,
            output_keep_prob=keep_out,
            state_keep_prob=keep_state
        )
# rnn_cell

def bidirect_cell(
        inputs,
        inputs_len,
        name,
        hidden_size=RNN_HIDDEN_SIZE,
        initial_state=None,
        keep_in=1.0,
        keep_out=1.0,
        keep_state=1.0
):
    return tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell(
                hidden_size,
                name=name+"_fw",
                keep_in=keep_in,
                keep_out=keep_out,
                keep_state=keep_state
            ),
            cell_bw=rnn_cell(
                hidden_size,
                name=name+"_bw",
                keep_in=keep_in,
                keep_out=keep_out,
                keep_state=keep_state
            ),
            inputs=inputs,
            sequence_length=inputs_len,
            initial_state_fw=initial_state,
            initial_state_bw=initial_state,
            dtype=tf.float32
        )
# bidirect_cell

'''
def batch_norm_layer(x, train_phase):
    bn_train = tf.batch_norm(x, decay=0.999, center=True, scale=True,
        is_training=True,
        reuse=None, # is this right?
        trainable=True
    )

    bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
            is_training=False,
        reuse=True, # is this right?
        trainable=True
    )
    return tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
#batch_norm_layer
'''

def group_outputs(outputs, mode=None, bn=False, is_training=False):
    if mode == "concat":
        outputs = tf.concat([outputs[0], outputs[1]], axis=2)
    elif mode == "sum":
        outputs = tf.add(outputs[0], outputs[1])
    else:
        raise
    if bn:
        outputs = tf.layers.batch_normalization(
                outputs,
                training=True,
                trainable=False
        )
    return outputs
# group_outputs

'''
def group_LSTM_state(states, make=False, mode=None, bn=False, is_training=False):
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
        states = batch_norm_layer(
                states,
                is_training
        )
    if make:
        return tf.contrib.rnn.LSTMStateTuple(states[0], states[1])
    return states
# make_LSTM_state
'''

class Model:
    def __init__(
            self,
            mode="train", # "train", "test" or "both"
            logs=True,
            batch_size=BATCH_SIZE,
            embedding_size=EMBEDDING_SIZE
    ):
        self.__setup_constants(
                bs=batch_size,
                es=embedding_size
        )
        self.__logs = logs
        self.__mode = mode
        self.__load_embeddings()
        self.__setup_model()
    # __init__

    def __setup_constants(self, bs, es):
        self.__BATCH_SIZE = bs
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
            self.__batch_size = tf.shape(self.question)[0]
            self.__question_size = tf.shape(self.question)[1]
            self.__context_size = tf.shape(self.context)[1]
            self.__context_with_features = tf.concat(
                    [
                        tf.nn.dropout(
                            self.context,
                            self.keep_prob
                        ),
                        self.context_float_features,
                        tf.cast(
                            self.context_int_features[:, :, 0:3],
                            tf.float32
                        ),
                        tf.one_hot(
                            self.context_int_features[:, :, 3],
                            TAG_NUM,
                            dtype=tf.float32
                        ),
                        tf.one_hot(
                            self.context_int_features[:, :, 4],
                            ENT_NUM,
                            dtype=tf.float32
                        )
                    ],
                    axis=-1
                )
            with tf.variable_scope("question"):
                question_out = self.__setup_question()
                question_features = self.__setup_question_features(
                        question_out
                )
            with tf.variable_scope("context"):
                context_out = self.__setup_context(
                        question_out,
                        question_features
                )
            with tf.variable_scope("loss"):
                points = self.__setup_loss(
                        question_out,
                        question_features,
                        context_out
                )
            with tf.variable_scope("answer"):
                self.__setup_answer(
                        points[0],
                        points[1]
                )
            self.__summary_all()
            '''
            optimizer = tf.train.RMSPropOptimizer(0.001)
            self.train_step = optimizer.minimize(self.loss)
            '''
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.RMSPropOptimizer(0.001)
                self.train_step = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
    # __setup_model

    def __setup_inputs(self):
        self.context = tf.placeholder(
            tf.float32,
            name="context",
            shape=(None, None, self.__EMBEDDING_SIZE)
        )
        self.context_len = tf.placeholder(
            tf.int32,
            name="context_lenght",
            shape=(None)
        )
        self.context_int_features = tf.placeholder(
            tf.int32,
            name="context_int_features",
            shape=(None, None, INT_FEATURES_SIZE)
        )
        self.context_float_features = tf.placeholder(
            tf.float32,
            name="context_float_features",
            shape=(None, None, FLOAT_FEATURES_SIZE)
        )
        self.question = tf.placeholder(
            tf.float32,
            name="question",
            shape=(None, None, self.__EMBEDDING_SIZE)
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
        self.answer_mask = tf.placeholder(
            tf.float32,
            name="answer_mask",
            shape=(None, None)
        )
        self.keep_prob = tf.placeholder(
            tf.float32,
            name="keep_prob"
        )
        self.is_training = tf.placeholder(
            tf.bool,
            name="is_training"
        )
    # __setup_inputs

    def __setup_question(self):
        question_layer1 = bidirect_cell(
                tf.nn.dropout(
                    self.question,
                    self.keep_prob
                ),
                self.question_len,
                name="q_rnn1",
                hidden_size=RNN_HIDDEN_SIZE,
                keep_state=self.keep_prob,
                keep_in=1.0
            )
        output1 = group_outputs(
                question_layer1[0],
                mode="concat",
                bn=False,
                is_training=self.is_training
        )
        question_layer2 = bidirect_cell(
                output1,
                self.question_len,
                name="q_rnn2",
                hidden_size=RNN_HIDDEN_SIZE,
                keep_state=self.keep_prob,
                keep_in=1.0
        )
        output2 = group_outputs(
                question_layer2[0],
                mode="concat",
                bn=False,
                is_training=self.is_training
        )
        return question_layer2
    # __setup_question

    def __setup_question_features(self, question_out):
        question_outputs = group_outputs(
                question_out[0],
                mode='concat',
                bn=False,
                is_training=self.is_training
        )
        question_attention = tf.layers.dense(
                inputs=question_outputs,
                units=1,
                use_bias=True,
                name='question_attention'
        )
        question_attention = tf.reshape(question_attention, (self.__batch_size, self.__question_size))
        question_mask = tf.sequence_mask(
                self.question_len,
                maxlen=self.__question_size,
                name='question_attention_mask',
                dtype=tf.float32
        )
        question_attention = tf.nn.softmax(question_attention) * question_mask
        '''
        for_divide = tf.reduce_sum(
                question_attention,
                axis=1
        )
        #TODO
        question_attention = question_attention / for_divide[:,tf.newaxis]
        '''
        question_features = tf.reduce_sum(
                question_outputs * question_attention[:,:,tf.newaxis],
                axis=1,
                name='question_features'
        )
        return question_features
    # __setup_question_features

    def __setup_context(self, question_out, question_features):
        context_layer1 = bidirect_cell(
                self.__context_with_features,
                self.context_len,
                name="c_rnn1",
                hidden_size=RNN_HIDDEN_SIZE,
                keep_state=self.keep_prob,
                keep_in=1.0
        )
        context_layer2 = bidirect_cell(
                group_outputs(
                    context_layer1[0],
                    mode="concat",
                    bn=False,
                    is_training=self.is_training
                ),
                self.context_len,
                name="c_rnn2",
                hidden_size=RNN_HIDDEN_SIZE,
                keep_state=self.keep_prob,
                keep_in=1.0
        )
        return context_layer2
    # __setup_context()

    def __setup_loss(self,
            question_out,
            question_features,
            context_out
    ):
        context_outputs = group_outputs(
                context_out[0],
                mode="concat",
                bn=False,
                is_training=self.is_training
        )
        dense_begin = tf.layers.dense(
                inputs=context_outputs,
                units=2*RNN_HIDDEN_SIZE,
                use_bias=True,
                name='dense_begin'
        )
        dense_end = tf.layers.dense(
                inputs=context_outputs,
                units=2*RNN_HIDDEN_SIZE,
                use_bias=True,
                name='dense_end'
        )
        question_features = tf.reshape(
                question_features,
                (
                    self.__batch_size,
                    2*RNN_HIDDEN_SIZE,
                    1
                )
        )
        points_begin = tf.matmul(
                dense_begin,
                question_features,
                name='points_begin'
        )
        points_end = tf.matmul(
                dense_end,
                question_features,
                name='points_end'
        )
        points_begin = tf.reshape(
                points_begin,
                (
                    self.__batch_size,
                    self.__context_size
                )
        )
        points_end = tf.reshape(
                points_end,
                (
                    self.__batch_size,
                    self.__context_size
                )
        )
        loss_begin = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.true_answer_begin,
                logits=points_begin,
                name="softmax_begin"
        )
        loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.true_answer_end,
                logits=points_end,
                name="softmax_end"
        )
        batch_loss_begin = tf.reduce_mean(loss_begin, name='loss_begin')
        batch_loss_end = tf.reduce_mean(loss_end, name='loss_end')
        self.loss = tf.add(batch_loss_begin, batch_loss_end, name='loss')
        return points_begin, points_end
    # __setup_loss_and_answers

    def __setup_answer(self, points_begin, points_end):
        context_mask = tf.sequence_mask(
                self.context_len,
                maxlen=self.__context_size,
                name='context_mask',
                dtype=tf.float32
        )
        softmax_begin = tf.nn.softmax(points_begin) * context_mask
        softmax_end = tf.nn.softmax(points_end) * context_mask
        matrix_answers = tf.matmul(
                tf.reshape(
                    softmax_begin,
                    (self.__batch_size, self.__context_size, 1)
                ),
                tf.reshape(
                    softmax_end,
                    (self.__batch_size, 1, self.__context_size)
                )
        )
        matrix_answers = tf.multiply(
                matrix_answers,
                self.answer_mask[tf.newaxis,:,:]
        )
        answer = tf.argmax(
                tf.reshape(
                    matrix_answers,
                    (
                        self.__batch_size,
                        self.__context_size * self.__context_size
                    )
                ),
                axis=-1,
                output_type=tf.int32
        )
        self.answer_begin = tf.cast(
                answer // self.__context_size,
                tf.int32
        )
        self.answer_end = tf.cast(
                answer % self.__context_size,
                tf.int32
        )
    # __setup_answers

    def __summary_all(self):
        tf.summary.scalar('Loss', self.loss)
        good_begin = tf.equal(self.true_answer_begin, self.answer_begin)
        good_end = tf.equal(self.true_answer_end, self.answer_end)
        tf.summary.scalar('Answer begin accuracy',
            tf.count_nonzero(
                good_begin,
                dtype=tf.float32
            ) / tf.cast(tf.size(good_begin), tf.float32)
        )
        tf.summary.scalar('Answer begin and end accuracy',
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
        self.F1_score = 2 * TP / (2 * TP + FPN)
        tf.summary.scalar('F1 score',
            tf.reduce_mean(
                self.F1_score
            )
        )
        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(
                LOGS_PATH + SEP + "curr/train",
                graph=self.graph
        )
        self.test_writer = tf.summary.FileWriter(
                LOGS_PATH + SEP + "curr/test",
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
            keep_prob=1.0,
            window=20,
            epochs=1,
            steps=None
    ):
        out = False
        step = 0
        mask = get_answer_mask(
                train[0][0].shape[1],
                window
        )
        for e in range(epochs):
            for batch in tqdm(
                    next_batch(
                        train, 
                        batch_size, 
                        self.__embeddings
                    )
            ):
                if train_summary_every != None and step % train_summary_every == 0:
                    summary, _ = session.run(
                            [
                                self.summary,
                                self.train_step
                            ],
                            {
                                self.context: batch[0][0],
                                self.context_len: batch[0][1],
                                self.context_int_features: batch[1][0],
                                self.context_float_features: batch[1][1],
                                self.question: batch[2][0],
                                self.question_len: batch[2][1],
                                self.true_answer_begin: batch[3][0],
                                self.true_answer_end: batch[3][1],
                                self.answer_mask: mask,
                                self.keep_prob: keep_prob,
                                self.is_training: True
                            }
                         )
                    self.train_writer.add_summary(summary, step)
                else:
                    session.run(
                            self.train_step,
                            {
                                self.context: batch[0][0],
                                self.context_len: batch[0][1],
                                self.context_int_features: batch[1][0],
                                self.context_float_features: batch[1][1],
                                self.question: batch[2][0],
                                self.question_len: batch[2][1],
                                self.true_answer_begin: batch[3][0],
                                self.true_answer_end: batch[3][1],
                                self.answer_mask: mask,
                                self.keep_prob: keep_prob,
                                self.is_training: True
                            }
                    )
                if test_summary_every != None and step % test_summary_every == 0:
                    self.evaluate(
                            session,
                            test,
                            window,
                            step,
                            keep_prob=keep_prob,
                            batch_size=BATCH_SIZE
                    )
                step += 1
                if steps != None and step == steps:
                    out = True
                    break
            if out:
                break
    # train_model

    def evaluate(self,
            session,
            test,
            window,
            x=None,
            keep_prob=1.0,
            batch_size=BATCH_SIZE,
            tensorboard=True
    ):
        mask = get_answer_mask(test[0][0].shape[1], window)
        if tensorboard:
            batch = get_random_batch(test, batch_size, self.__embeddings)
            summary = session.run(
                self.summary,
                {
                    self.context: batch[0][0],
                    self.context_len: batch[0][1],
                    self.context_int_features: batch[1][0],
                    self.context_float_features: batch[1][1],
                    self.question: batch[2][0],
                    self.question_len: batch[2][1],
                    self.true_answer_begin: batch[3][0],
                    self.true_answer_end: batch[3][1],
                    self.answer_mask: mask,
                    self.keep_prob: keep_prob,
                    self.is_training: False
                }
            )
            self.test_writer.add_summary(summary, x)
        else:
            N = 0
            f1_sum = 0
            for batch in tqdm(next_batch(
                    test,
                    batch_size,
                    self.__embeddings
                )
            ):
                F1_score = session.run(
                    [
                        self.F1_score
                    ],
                    {
                        self.context: batch[0][0],
                        self.context_len: batch[0][1],
                        self.context_int_features: batch[1][0],
                        self.context_float_features: batch[1][1],
                        self.question: batch[2][0],
                        self.question_len: batch[2][1],
                        self.true_answer_begin: batch[3][0],
                        self.true_answer_end: batch[3][1],
                        self.answer_mask: mask,
                        self.keep_prob: keep_prob,
                        self.is_training: False
                    }
                )
                f1_sum += F1_score[0].sum()
                N += batch[0][0].shape[0]
            print("F1 score for all test =", f1_sum / N)
            return f1_sum / N

    # evaluate

    def get_answer(self, session, data, window=15, keep_prob=1.0):
        mask = get_answer_mask(data[0][0].shape[1], window)
        batch = get_random_batch(data, 1, self.__embeddings)
        answer_begin, answer_end = session.run(
                    [
                        self.answer_begin,
                        self.answer_end
                    ],
                    {
                        self.context: batch[0][0],
                        self.context_len: batch[0][1],
                        self.context_int_features: batch[1][0],
                        self.context_float_features: batch[1][1],
                        self.question: batch[2][0],
                        self.question_len: batch[2][1],
                        self.true_answer_begin: batch[3][0],
                        self.true_answer_end: batch[3][1],
                        self.answer_mask: mask,
                        self.keep_prob: keep_prob,
                        self.is_training: False
                    }
                )
        return answer_begin, answer_end
    # get_answer

    def save_model(self, session, path=MODEL_PATH, add=''):
        self.saver.save(session, path + add)
    # save_movel

    def load_model(self, session, path=MODEL_PATH):
        self.saver.restore(session, path)
    # load_model

# Model

