#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jan-06-20 11:47
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import sys

import tensorflow as tf

from data_utils import *
from seq2seq_model import *

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_integer(
    "batch_size", 64, "Batch size to use during training.")  # 32 64 256 大小根据机器选择
tf.app.flags.DEFINE_integer(
    "numEpochs", 30, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("en_de_seq_len", 20, "English vocabulary size.")
tf.app.flags.DEFINE_integer(
    "max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer(
    "steps_per_checkpoint", 100, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string(
    "train_dir", './model', "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string(
    "tmp", './tmp', "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer(
    "beam_size", 5, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean(
    "beam_search", True, "Set to True for beam_search.")
tf.app.flags.DEFINE_boolean(
    "decode", True, "Set to True for interactive decoding.")
FLAGS = tf.app.flags.FLAGS


def create_model(session, forward_only, beam_search, beam_size=5):
    """Create translation model and initialize or load parameters in session."""
    model = Seq2SeqModel(
        FLAGS.en_vocab_size, FLAGS.en_vocab_size, [10, 10],
        FLAGS.size, FLAGS.num_layers, FLAGS.batch_size,
        FLAGS.learning_rate, forward_only=forward_only, beam_search=beam_search, beam_size=beam_size)
    ckpt = tf.train.latest_checkpoint(FLAGS.train_dir)
    model_path = os.path.join(
        FLAGS.tmp, "chat_bot.ckpt-0")
    if forward_only:
        model.saver.restore(session, model_path)
    elif ckpt and tf.gfile.Exists(ckpt + ".meta"):
        print("Reading model parameters from checkpoint %s" % ckpt)
        model.saver.restore(session, ckpt)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def decode():
    with tf.Session() as sess:  # 打开tensorflow session需要时间
        beam_size = FLAGS.beam_size
        beam_search = FLAGS.beam_search
        model = create_model(
            sess, forward_only=True, beam_search=beam_search, beam_size=beam_size)
        model.batch_size = 1
        data_path = 'data/dataset-cornell-length10-filter1-vocabSize40000.pkl'
        data_path = os.path.join(os.path.abspath("."), data_path)
        word2id, id2word, trainingSamples = loadDataset(data_path)

        if beam_search:
            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            while sentence:
                batch = sentence2enco(sentence, word2id, model.en_de_seq_len)
                beam_path, beam_symbol = model.step(sess, batch.encoderSeqs, batch.decoderSeqs, batch.targetSeqs,
                                                    batch.weights, goToken)
                paths = [[] for _ in range(beam_size)]
                curr = [i for i in range(beam_size)]
                num_steps = len(beam_path)
                for i in range(num_steps-1, -1, -1):
                    for kk in range(beam_size):
                        paths[kk].append(beam_symbol[i][curr[kk]])
                        curr[kk] = beam_path[i][curr[kk]]
                recos = set()
                print("Replies --------------------------------------->")
                for kk in range(beam_size):
                    foutputs = [int(logit) for logit in paths[kk][::-1]]
                    if eosToken in foutputs:
                        foutputs = foutputs[:foutputs.index(eosToken)]
                    rec = " ".join([tf.compat.as_str(id2word[output])
                                    for output in foutputs if output in id2word])
                    if rec not in recos:
                        recos.add(rec)
                        print(rec)
                print("> ", "")
                sys.stdout.flush()
                sentence = sys.stdin.readline()
                # return recos


def main():
    decode()


if __name__ == "__main__":
    main()
