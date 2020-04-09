#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-09-20 21:03
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import sys
import math
import time
import tensorflow as tf
from data_utils import *
from model import *
from tqdm import tqdm


def beam_search(sess, sentence, word2id, id2word, model, beam_size=5):
    if sentence:
        batch = sentence2enco(sentence, word2id, model.en_de_seq_len)
        beam_path, beam_symbol = model.step(sess, batch.encoderSeqs, batch.decoderSeqs, batch.targetSeqs,
                                            batch.weights, goToken)
        paths = [[] for _ in range(beam_size)]
        indices = [i for i in range(beam_size)]
        num_steps = len(beam_path)
        for i in reversed(range(num_steps)):
            for kk in range(beam_size):
                paths[kk].append(beam_symbol[i][indices[kk]])
                indices[kk] = beam_path[i][indices[kk]]

        recos = []
        for kk in range(beam_size):
            foutputs = [int(logit) for logit in paths[kk][::-1]]
            if eosToken in foutputs:
                foutputs = foutputs[:foutputs.index(eosToken)]
            rec = " ".join([tf.compat.as_str(id2word[output])
                            for output in foutputs if output in id2word])
            if rec not in recos:
                recos.append(rec)
        return recos


def main():
    pass
    # with tf.Session() as sess:
    #     beam_size = 5
    #     if_beam_search = True
    #     model = create_model(
    #         sess, True, beam_search=if_beam_search, beam_size=beam_size)
    #     model.batch_size = 1
    #     data_path = DATA_PATH
    #     word2id, id2word, trainingSamples = load_dataset(data_path)

    #     sys.stdout.write("> ")
    #     sys.stdout.flush()
    #     sentence = sys.stdin.readline()
    #     while sentence:
    #         recos = beam_search(sess, sentence=sentence, word2id=word2id,
    #                             id2word=id2word, model=model)
    #         print("Replies --------------------------------------->")
    #         print(recos)
    #         sys.stdout.write("> ")
    #         sys.stdout.flush()
    #         sentence = sys.stdin.readline()


if __name__ == "__main__":
    main()
